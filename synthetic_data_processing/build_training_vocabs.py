import argparse
import glob
import json
import pyarrow.parquet as pq
from collections import Counter
from pathlib import Path

def build_vocabs(parquet_dir, output_dir, top_k_ret=1024):
    """
    Scans Parquet files to build vocabularies for:
    - comm (Full vocab, it's small)
    - ret (Top-K, to handle the long tail of sizes)
    """
    comm_counter = Counter()
    ret_counter = Counter()
    
    # Recursively find all parquet files
    files = sorted(glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True))
    print(f"[INFO] Scanning {len(files)} files in {parquet_dir}...")
    
    if not files:
        print("[WARN] No parquet files found!")
        return

    for fpath in files:
        try:
            # Read only relevant columns to save memory
            t = pq.read_table(fpath, columns=['comm', 'ret'])
            df = t.to_pandas()
            
            # Comm: Convert to string, drop NA
            comms = df['comm'].dropna().astype(str).tolist()
            comm_counter.update(comms)
            
            # Ret: Drop NA
            rets = df['ret'].dropna().tolist()
            ret_counter.update(rets)
            
            print(f"   Scanned {Path(fpath).name}: {len(comms)} lines", end='\r')
        except Exception as e:
            print(f"\n[WARN] Failed to read {fpath}: {e}")

    print("\n[INFO] Scan complete.")
    
    # --- Build Comm Vocab (Full) ---
    # Reserve 0 for PAD, 1 for UNK
    comm_vocab = {"<PAD>": 0, "<UNK>": 1}
    # Sort by frequency for determinism
    for name, count in comm_counter.most_common():
        comm_vocab[name] = len(comm_vocab)
        
    # --- Build Ret Vocab (Top-K) ---
    ret_vocab = {"<PAD>": 0, "<UNK>": 1}
    
    # Get Top-K most frequent values
    most_common_ret = ret_counter.most_common(top_k_ret)
    
    # Calculate coverage
    coverage_count = sum(c for _, c in most_common_ret)
    total_count = sum(ret_counter.values())
    coverage_pct = (coverage_count / total_count * 100) if total_count > 0 else 0.0
    
    print(f"[INFO] Ret Metadata:")
    print(f"       Total Unique Values: {len(ret_counter)}")
    print(f"       Top-{top_k_ret} Coverage: {coverage_pct:.2f}%")
    
    for val, _ in most_common_ret:
        # Convert to string for JSON key
        ret_vocab[str(int(val))] = len(ret_vocab)

    # --- Save ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    out_comm = Path(output_dir) / "vocab_comm.json"
    out_ret = Path(output_dir) / "vocab_ret.json"
    
    with open(out_comm, "w") as f:
        json.dump(comm_vocab, f, indent=2)
        
    with open(out_ret, "w") as f:
        json.dump(ret_vocab, f, indent=2)
        
    print(f"[SUCCESS] Vocabs saved to {output_dir}")
    print(f"          - vocab_comm.json ({len(comm_vocab)} tokens)")
    print(f"          - vocab_ret.json ({len(ret_vocab)} tokens)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True, help="Root directory of parquet files")
    parser.add_argument("--output-dir", required=True, help="Where to save json vocabs")
    parser.add_argument("--top-k-ret", type=int, default=1024, help="Max vocab size for ret")
    args = parser.parse_args()
    
    build_vocabs(args.parquet_dir, args.output_dir, args.top_k_ret)
