#!/usr/bin/env python3
"""
Improved Parquet to Windowed NPZ Converter

Key Improvements:
1. **Smaller Shards**: Breaks down large files into smaller NPZ shards (configurable windows per shard)
2. **Better GPU Utilization**: Smaller files load faster, reducing data loading bottleneck
3. **Memory Efficient**: Processes and saves in chunks to avoid OOM
4. **Parallel Processing**: Supports multiprocessing for faster conversion
5. **Progress Tracking**: Better visibility into conversion progress

Usage:
    python parquet_to_windowed_npz_improved.py \
        --input-dir dataset/parquet \
        --output-dir dataset/windowed_npz_1024_improved \
        --vocab-dir dataset/metadata_all_events \
        --seq-len 1024 \
        --stride 512 \
        --windows-per-shard 1000 \
        --workers 8
"""

import argparse
import glob
import json
import os
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_vocab(path):
    """Load vocabulary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def process_file_to_shards(
    parquet_path: str,
    output_dir: str,
    vocab_comm: dict,
    vocab_ret: dict,
    seq_len: int = 1024,
    stride: int = 512,
    windows_per_shard: int = 1000,
    tid_buckets: int = 256,
    fd_cap: int = 1024,
    split: str = "train",
    benchmark: str = None  # NEW: benchmark name
):
    """
    Reads a parquet file, tokenizes/normalizes columns, sequences them, 
    and saves as MULTIPLE smaller NPZ shards.
    
    Key Improvement: Instead of saving all windows in one large NPZ file,
    this function creates multiple smaller shards with `windows_per_shard` windows each.
    This dramatically improves data loading speed during training.
    
    Args:
        parquet_path: Path to input parquet file
        output_dir: Output directory for NPZ shards
        vocab_comm: Command vocabulary mapping
        vocab_ret: Return value vocabulary mapping
        seq_len: Sequence length for windows
        stride: Stride for sliding window
        windows_per_shard: Number of windows per NPZ shard (DEFAULT: 1000)
        tid_buckets: Number of buckets for thread ID hashing
        fd_cap: Maximum file descriptor value
        split: Dataset split (train/val/test)
        benchmark: Benchmark name (e.g., 'pybench', 'ffmpeg')
    
    Returns:
        Number of shards created
    """
    try:
        table = pq.read_table(parquet_path)
    except Exception as e:
        print(f"[ERR] Failed to read {parquet_path}: {e}")
        return 0

    # Convert to pandas for easier mapping
    df = table.to_pandas()
    N = len(df)
    
    if N < seq_len:
        print(f"[WARN] File {Path(parquet_path).name} too short ({N} < {seq_len}). Skipping.")
        return 0

    # --- 1. Tokenization & Normalization ---
    
    # Event ID (Identity)
    event_arr = df['event_id'].fillna(0).astype('int32').values

    # CPU (Identity)
    cpu_arr = df['cpu_id'].fillna(-1).astype('int8').values
    
    # Delta (Log Norm)
    dt_arr = df['delta_s'].fillna(0.0).astype('float32').values
    dt_arr = np.log1p(dt_arr)  # log(1 + x)

    # TID (Hash)
    tid_raw = df['tid'].fillna(0).astype('int64').values
    tid_arr = (tid_raw % tid_buckets).astype('int16')

    # FD (Clamp)
    fd_raw = df['fd'].fillna(-1).astype('int64').values
    fd_arr = np.where((fd_raw < 0) | (fd_raw >= fd_cap), fd_cap, fd_raw).astype('int16')

    # Comm (Vocab)
    comm_unk = vocab_comm.get("<UNK>", 1)
    
    def map_comm(x):
        return vocab_comm.get(str(x), comm_unk)
    
    comm_arr = df['comm'].apply(map_comm).astype('int16').values

    # Ret (Vocab)
    ret_unk = vocab_ret.get("<UNK>", 1)
    
    def map_ret(x):
        try:
            k = str(int(x))
            return vocab_ret.get(k, ret_unk)
        except:
            return ret_unk
            
    ret_arr = df['ret'].apply(map_ret).astype('int16').values

    # --- 2. Windowing ---
    
    starts = np.arange(0, N - seq_len + 1, stride)
    if len(starts) == 0:
        return 0

    n_wins = len(starts)
    
    # --- 3. Save in Smaller Shards ---
    
    stub = Path(parquet_path).stem
    shard_count = 0
    
    # Process windows in chunks
    for chunk_start in range(0, n_wins, windows_per_shard):
        chunk_end = min(chunk_start + windows_per_shard, n_wins)
        chunk_size = chunk_end - chunk_start
        
        # Allocate buffers for this shard only
        out_event = np.empty((chunk_size, seq_len), dtype='int32')
        out_dt    = np.empty((chunk_size, seq_len), dtype='float32')
        out_cpu   = np.empty((chunk_size, seq_len), dtype='int8')
        out_tid   = np.empty((chunk_size, seq_len), dtype='int16')
        out_fd    = np.empty((chunk_size, seq_len), dtype='int16')
        out_comm  = np.empty((chunk_size, seq_len), dtype='int16')
        out_ret   = np.empty((chunk_size, seq_len), dtype='int16')
        
        # Fill buffers
        for i, global_idx in enumerate(range(chunk_start, chunk_end)):
            s = starts[global_idx]
            e = s + seq_len
            out_event[i] = event_arr[s:e]
            out_dt[i]    = dt_arr[s:e]
            out_cpu[i]   = cpu_arr[s:e]
            out_tid[i]   = tid_arr[s:e]
            out_fd[i]    = fd_arr[s:e]
            out_comm[i]  = comm_arr[s:e]
            out_ret[i]   = ret_arr[s:e]
        
        # Save this shard
        out_name = f"{stub}_L{seq_len}_S{stride}_shard{shard_count:04d}.npz"
        
        # NEW: Organize by benchmark/split instead of split only
        if benchmark:
            out_path = os.path.join(output_dir, benchmark, split, out_name)
        else:
            out_path = os.path.join(output_dir, split, out_name)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        np.savez_compressed(
            out_path,
            event=out_event,
            dt=out_dt,
            cpu=out_cpu,
            tid=out_tid,
            fd=out_fd,
            comm=out_comm,
            ret=out_ret
        )
        
        shard_count += 1
    
    return shard_count


def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    parquet_path, output_dir, vocab_comm, vocab_ret, seq_len, stride, windows_per_shard, split, benchmark = args
    try:
        shard_count = process_file_to_shards(
            parquet_path, output_dir, vocab_comm, vocab_ret,
            seq_len, stride, windows_per_shard, split=split, benchmark=benchmark
        )
        return (parquet_path, shard_count, None)
    except Exception as e:
        return (parquet_path, 0, str(e))


def extract_benchmark_name(parquet_path):
    """
    Extract benchmark name from parquet file path.
    
    Assumes structure like: .../enriched_parquet/pybench/file.parquet
    or .../enriched_parquet/txt_traces_pybench/file.parquet
    
    Returns benchmark name (e.g., 'pybench', 'ffmpeg', 'scimark2')
    """
    path_parts = Path(parquet_path).parts
    
    # Look for common benchmark names in the path
    benchmark_names = ['pybench', 'ffmpeg', 'scimark2', 'stream', 'unpack-linux', 
                      'compress-gzip', 'iozone', 'ramspeed', 'phpbench']
    
    for part in reversed(path_parts):
        # Check if this part contains a benchmark name
        part_lower = part.lower()
        for bench in benchmark_names:
            if bench in part_lower:
                return bench
    
    # Fallback: use parent directory name
    return Path(parquet_path).parent.name


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to smaller NPZ shards for improved training performance"
    )
    parser.add_argument("--input-dir", required=True, help="Dir containing .parquet files (recursive)")
    parser.add_argument("--output-dir", required=True, help="Output root for npz dataset")
    parser.add_argument("--vocab-dir", required=True, help="Dir containing vocab_comm.json, etc")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length for windows")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")
    parser.add_argument("--windows-per-shard", type=int, default=1000, 
                       help="Number of windows per NPZ shard (smaller = faster loading)")
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1", help="Train,Val,Test ratio")
    parser.add_argument("--workers", type=int, default=None, 
                       help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args()
    
    # Set workers
    if args.workers is None:
        args.workers = cpu_count()
    
    print(f"[INFO] Configuration:")
    print(f"       Sequence Length: {args.seq_len}")
    print(f"       Stride: {args.stride}")
    print(f"       Windows per Shard: {args.windows_per_shard}")
    print(f"       Workers: {args.workers}")
    
    # Load Vocabs
    vocab_comm = load_vocab(os.path.join(args.vocab_dir, "vocab_comm.json"))
    vocab_ret = load_vocab(os.path.join(args.vocab_dir, "vocab_ret.json"))
    
    # Find Files
    files = sorted(glob.glob(f"{args.input_dir}/**/*.parquet", recursive=True))
    
    # Parse ratios
    r_train, r_val, r_test = map(float, args.split_ratio.split(","))
    total_r = r_train + r_val + r_test
    r_train /= total_r
    r_val /= total_r
    
    # Deterministic split
    rng = np.random.RandomState(42)
    rng.shuffle(files)
    
    n_files = len(files)
    n_train = int(n_files * r_train)
    n_val = int(n_files * r_val)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]
    
    print(f"[INFO] Found {n_files} parquet files.")
    print(f"       Train: {len(train_files)}")
    print(f"       Val:   {len(val_files)}")
    print(f"       Test:  {len(test_files)}")
    
    # Process each split
    def process_split(file_list, split_name):
        print(f"\n--- Processing {split_name} ({len(file_list)} files) ---")
        
        if args.workers > 1:
            # Parallel processing
            task_args = [
                (f, args.output_dir, vocab_comm, vocab_ret, 
                 args.seq_len, args.stride, args.windows_per_shard, split_name,
                 extract_benchmark_name(f))  # NEW: extract benchmark name
                for f in file_list
            ]
            
            with Pool(processes=args.workers) as pool:
                results = list(tqdm(
                    pool.imap(process_file_wrapper, task_args),
                    total=len(file_list),
                    desc=f"{split_name}"
                ))
            
            # Summary
            total_shards = sum(r[1] for r in results)
            errors = [r for r in results if r[2] is not None]
            
            print(f"[{split_name}] Created {total_shards} shards from {len(file_list)} files")
            if errors:
                print(f"[{split_name}] {len(errors)} files had errors:")
                for path, _, err in errors[:5]:  # Show first 5 errors
                    print(f"  - {Path(path).name}: {err}")
        else:
            # Sequential processing
            total_shards = 0
            for f in tqdm(file_list, desc=f"{split_name}"):
                benchmark = extract_benchmark_name(f)  # NEW: extract benchmark name
                shard_count = process_file_to_shards(
                    f, args.output_dir, vocab_comm, vocab_ret,
                    args.seq_len, args.stride, args.windows_per_shard,
                    split=split_name, benchmark=benchmark
                )
                total_shards += shard_count
            
            print(f"[{split_name}] Created {total_shards} shards from {len(file_list)} files")
    
    process_split(train_files, "train")
    process_split(val_files, "val")
    process_split(test_files, "test")
    
    print("\n[SUCCESS] Conversion complete!")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Output structure: {args.output_dir}/{{benchmark}}/{{train|val|test}}/")
    print(f"[TIP] Use the improved dataset loader for best performance")


if __name__ == "__main__":
    main()
