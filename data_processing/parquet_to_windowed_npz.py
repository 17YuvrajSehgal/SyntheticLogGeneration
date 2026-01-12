#!/usr/bin/env python3
import argparse
import glob
import json
import os
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

def load_vocab(path):
    with open(path, 'r') as f:
        return json.load(f)

def process_file(
    parquet_path: str,
    output_dir: str,
    vocab_comm: dict,
    vocab_ret: dict,
    seq_len: int = 1024,
    stride: int = 512,
    tid_buckets: int = 256,
    fd_cap: int = 1024,
    split: str = "train"
):
    """
    Reads a parquet file, tokenizes/normalizes columns, sequences them, and saves as NPZ.
    """
    try:
        table = pq.read_table(parquet_path)
    except Exception as e:
        print(f"[ERR] Failed to read {parquet_path}: {e}")
        return

    # Convert to pandas for easier mapping
    df = table.to_pandas()
    N = len(df)
    
    if N < seq_len:
        print(f"[WARN] File {Path(parquet_path).name} too short ({N} < {seq_len}). Skipping.")
        return

    # --- 1. Tokenization & Normalization ---

    # Event ID (Identity)
    # Ensure int32
    event_arr = df['event_id'].fillna(0).astype('int32').values

    # CPU (Identity)
    cpu_arr = df['cpu_id'].fillna(-1).astype('int8').values
    
    # Delta (Log Norm)
    # log(dt + epsilon) or log1p(dt)
    dt_arr = df['delta_s'].fillna(0.0).astype('float32').values
    dt_arr = np.log1p(dt_arr) # log(1 + x) matches typical practice

    # TID (Hash)
    # tid % buckets
    tid_raw = df['tid'].fillna(0).astype('int64').values
    tid_arr = (tid_raw % tid_buckets).astype('int16')

    # FD (Clamp)
    # 0..cap, anything > cap -> cap (or cap-1). Let's explicitly cap.
    fd_raw = df['fd'].fillna(-1).astype('int64').values
    # Replace -1 (null) with 0 or distinct? Let's treat standard FDs.
    # We map missing/-1 to 0 (stdin/invalid) or max? 
    # Usually FD -1 implies not relevant. Let's map to `fd_cap` (as "None").
    fd_arr = np.where((fd_raw < 0) | (fd_raw >= fd_cap), fd_cap, fd_raw).astype('int16')

    # Comm (Vocab)
    # Map string -> ID. 
    # Pandas map is fast enough.
    comm_unk = vocab_comm.get("<UNK>", 1)
    comm_pad = vocab_comm.get("<PAD>", 0)
    
    def map_comm(x):
        return vocab_comm.get(str(x), comm_unk)
    
    comm_arr = df['comm'].apply(map_comm).astype('int16').values

    # Ret (Vocab)
    ret_unk = vocab_ret.get("<UNK>", 1)
    
    def map_ret(x):
        # x is int or float. Convert to int str.
        try:
            k = str(int(x))
            return vocab_ret.get(k, ret_unk)
        except:
            return ret_unk
            
    ret_arr = df['ret'].apply(map_ret).astype('int16').values

    # --- 2. Windowing ---
    
    # We want to create standard windows. 
    # Shape: (NumWindows, SeqLen)
    
    # Simple stride approach
    starts = np.arange(0, N - seq_len + 1, stride)
    if len(starts) == 0:
         return

    # We can use simple indexing or stride_tricks
    # For memory safety with large files, standard indexing loop is safer than stride_tricks for saving.
    # But usually we want to construct the big arrays and save.
    
    # Pre-allocate output buffers
    n_wins = len(starts)
    
    out_event = np.empty((n_wins, seq_len), dtype='int32')
    out_dt    = np.empty((n_wins, seq_len), dtype='float32')
    out_cpu   = np.empty((n_wins, seq_len), dtype='int8')
    out_tid   = np.empty((n_wins, seq_len), dtype='int16')
    out_fd    = np.empty((n_wins, seq_len), dtype='int16')
    out_comm  = np.empty((n_wins, seq_len), dtype='int16')
    out_ret   = np.empty((n_wins, seq_len), dtype='int16')
    
    for i, s in enumerate(starts):
        e = s + seq_len
        out_event[i] = event_arr[s:e]
        out_dt[i]    = dt_arr[s:e]
        out_cpu[i]   = cpu_arr[s:e]
        out_tid[i]   = tid_arr[s:e]
        out_fd[i]    = fd_arr[s:e]
        out_comm[i]  = comm_arr[s:e]
        out_ret[i]   = ret_arr[s:e]

    # --- 3. Save ---
    stub = Path(parquet_path).stem
    out_name = f"{stub}_L{seq_len}_S{stride}.npz"
    out_path = os.path.join(output_dir, split, out_name)
    
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
    
    print(f"[OK] Saved {out_path} ({n_wins} windows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Dir containing .parquet files (recursive)")
    parser.add_argument("--output-dir", required=True, help="Output root for npz dataset")
    parser.add_argument("--vocab-dir", required=True, help="Dir containing vocab_comm.json, etc")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1", help="Train,Val,Test ratio")
    
    args = parser.parse_args()
    
    # Load Vocabs
    vocab_comm = load_vocab(os.path.join(args.vocab_dir, "vocab_comm.json"))
    vocab_ret = load_vocab(os.path.join(args.vocab_dir, "vocab_ret.json"))
    
    # Prepare Dirs
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, s), exist_ok=True)
        
    # Find Files
    files = sorted(glob.glob(f"{args.input_dir}/**/*.parquet", recursive=True))
    
    # Shuffle files to distribute datasets across splits? 
    # Or keep runs contiguous? 
    # Usually better to split by Run ID to avoid data leakage.
    # We will simply assign files to splits deterministically.
    
    # Parse ratios
    r_train, r_val, r_test = map(float, args.split_ratio.split(","))
    # Normalize
    total_r = r_train + r_val + r_test
    r_train /= total_r
    r_val /= total_r
    
    # Deterministic split based on hash of filename or just index?
    # Index is fine if sorted.
    # Actually, we should probably group by Benchmark to ensure all benchmarks are in all splits, 
    # OR split by benchmark. Usually we want Generalization, so we train on 8 benchmarks test on 1?
    # User likely wants "In-Distribution" split (Train on 80% run of Mysql, Test on 20% run of Mysql).
    # So purely random file assignment is okay if files are "Runs".
    
    # We'll use a simple deterministic P-RNG
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
    
    # Process
    # (In a real scenario, this loop receives one file from SLURM array, 
    # but here we implement the looper. We can also make this script process ONE file if needed for SLURM).
    
    # To be SLURM friendly, let's allow processing specific files if they are passed?
    # For now, let's just loop. The user can run this as a big job, or we create a generator.
    
    # Helper
    def run_batch(file_list, split_name):
        print(f"--- Processing {split_name} ---")
        for f in file_list:
            process_file(
                f, 
                args.output_dir, 
                vocab_comm, 
                vocab_ret, 
                seq_len=args.seq_len, 
                stride=args.stride,
                split=split_name
            )
            
    run_batch(train_files, "train")
    run_batch(val_files, "val")
    run_batch(test_files, "test")

if __name__ == "__main__":
    main()
