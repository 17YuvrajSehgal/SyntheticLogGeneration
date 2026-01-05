#!/usr/bin/env python3
"""
Window + shard generator for LTTng all-events Parquet runs.

Creates .npz shards containing:
  event: int32 [N, L]
  dt:    uint8 [N, L]   (bucketed from dt_sec)
  cpu:   uint8 [N, L]

Key feature: correct sliding windows across Arrow batch boundaries
(using a carry buffer of last (window-1) rows).
"""

import os
import re
import argparse
import numpy as np
import pyarrow.parquet as pq

# -----------------------------
# DT bucketing
# -----------------------------
def dt_to_bucket(dt_sec: np.ndarray, num_buckets=256) -> np.ndarray:
    """
    seconds -> log1p(ns) -> bucket [0, num_buckets-1]
    """
    dt_sec = np.asarray(dt_sec, dtype=np.float64)
    dt_ns = np.maximum(dt_sec * 1e9, 0.0)
    log_dt = np.log1p(dt_ns)

    # log1p(1s) ~ 20.7; cap slightly above typical kernel deltas
    max_log = 22.0
    bucket = np.floor((log_dt / max_log) * num_buckets)
    return np.clip(bucket, 0, num_buckets - 1).astype(np.uint8)

# -----------------------------
# Utilities
# -----------------------------
def save_shard(events, dts, cpus, out_dir, split, run_id, shard_idx):
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    path = os.path.join(out_dir, split, f"run{run_id:02d}_shard{shard_idx:04d}.npz")
    np.savez_compressed(
        path,
        event=np.stack(events),
        dt=np.stack(dts),
        cpu=np.stack(cpus),
    )
    print(f"[WRITE] {path} ({len(events)} samples)")

def parse_run_id(filename: str) -> int:
    m = re.search(r"run(\d+)\.parquet$", filename)
    if not m:
        raise ValueError(f"Could not parse run id from filename: {filename}")
    return int(m.group(1))

# -----------------------------
# Core windowing with carry
# -----------------------------
def emit_windows_from_buffer(ev, dtb, cpu, window, stride, start_offset=0):
    """
    Emit windows from arrays starting at start_offset (inclusive).
    Returns number of windows emitted and the next start index to continue from.
    """
    n = len(ev)
    i = start_offset
    count = 0
    while i + window <= n:
        yield ev[i:i+window], dtb[i:i+window], cpu[i:i+window]
        count += 1
        i += stride
    return count, i

def process_parquet(
    parquet_path,
    out_dir,
    split,
    run_id,
    window,
    stride,
    shard_size,
    num_dt_buckets,
    event_col,
    dt_col,
    cpu_col,
    batch_size,
):
    print(f"[INFO] Processing {os.path.basename(parquet_path)}")

    pf = pq.ParquetFile(parquet_path)

    # carry buffers keep last (window-1) rows to not lose cross-batch windows
    carry_ev = np.empty((0,), dtype=np.int32)
    carry_dt = np.empty((0,), dtype=np.uint8)
    carry_cpu = np.empty((0,), dtype=np.uint8)

    # To avoid duplicate windows, we need to remember where the next window start should be
    # in the concatenated (carry + new_batch) buffer.
    next_start = 0

    shard_events, shard_dts, shard_cpus = [], [], []
    shard_idx = 0
    total_samples = 0

    cols = [event_col, dt_col, cpu_col]

    for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
        ev_new = batch.column(event_col).to_numpy(zero_copy_only=False).astype(np.int32)
        dt_new = batch.column(dt_col).to_numpy(zero_copy_only=False).astype(np.float64)
        cpu_new = batch.column(cpu_col).to_numpy(zero_copy_only=False).astype(np.uint16)

        # bucketize dt
        dtb_new = dt_to_bucket(dt_new, num_buckets=num_dt_buckets)

        # cpu to uint8 if possible (cpu ids usually small)
        # if cpu can exceed 255, switch to uint16 everywhere; for this dataset uint8 is fine.
        cpu_new = cpu_new.astype(np.uint8)

        # concatenate carry + new
        ev = np.concatenate([carry_ev, ev_new])
        dtb = np.concatenate([carry_dt, dtb_new])
        cpu = np.concatenate([carry_cpu, cpu_new])

        # Emit windows starting from next_start
        # We’ll compute windows and also compute the next_start that would be used if we had more data.
        i = next_start
        n = len(ev)

        while i + window <= n:
            shard_events.append(ev[i:i+window])
            shard_dts.append(dtb[i:i+window])
            shard_cpus.append(cpu[i:i+window])
            total_samples += 1
            i += stride

            if len(shard_events) >= shard_size:
                save_shard(shard_events, shard_dts, shard_cpus, out_dir, split, run_id, shard_idx)
                shard_idx += 1
                shard_events.clear(); shard_dts.clear(); shard_cpus.clear()

        # Now i is the next window start that needs more data (i+window > n)
        next_start = i

        # Update carry to last (window-1) elements
        keep = window - 1
        if n > keep:
            carry_ev = ev[-keep:]
            carry_dt = dtb[-keep:]
            carry_cpu = cpu[-keep:]
            # next_start was in old buffer coordinates; shift it to new carry coordinates
            # new buffer starts at n-keep
            next_start = max(0, next_start - (n - keep))
        else:
            # buffer smaller than window-1: keep it all
            carry_ev, carry_dt, carry_cpu = ev, dtb, cpu
            next_start = 0

    # flush remaining shard samples
    if shard_events:
        save_shard(shard_events, shard_dts, shard_cpus, out_dir, split, run_id, shard_idx)

    print(f"[INFO] Total samples from run {run_id}: {total_samples}")
    print(f"[INFO] Used columns: event='{event_col}', dt='{dt_col}', cpu='{cpu_col}'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--benchmark", required=True)

    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--shard_size", type=int, default=100_000)
    ap.add_argument("--dt_buckets", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1_000_000)

    ap.add_argument("--event_col", default="event_id")
    ap.add_argument("--dt_col", default="dt_sec")
    ap.add_argument("--cpu_col", default="cpu")

    ap.add_argument("--train_max_run", type=int, default=23)
    ap.add_argument("--val_max_run", type=int, default=27)

    args = ap.parse_args()

    parquet_files = sorted([f for f in os.listdir(args.parquet_dir) if f.endswith(".parquet")])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {args.parquet_dir}")

    base_out = os.path.join(args.out_dir, args.benchmark)

    for fname in parquet_files:
        run_id = parse_run_id(fname)
        if run_id <= args.train_max_run:
            split = "train"
        elif run_id <= args.val_max_run:
            split = "val"
        else:
            split = "test"

        process_parquet(
            parquet_path=os.path.join(args.parquet_dir, fname),
            out_dir=base_out,
            split=split,
            run_id=run_id,
            window=args.window,
            stride=args.stride,
            shard_size=args.shard_size,
            num_dt_buckets=args.dt_buckets,
            event_col=args.event_col,
            dt_col=args.dt_col,
            cpu_col=args.cpu_col,
            batch_size=args.batch_size,
        )

if __name__ == "__main__":
    main()