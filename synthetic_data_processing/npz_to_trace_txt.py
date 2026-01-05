#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np

MAX_LOG = 22.0  # must match make_window_shards.py :contentReference[oaicite:1]{index=1}

def bucket_to_dt_seconds(buckets: np.ndarray, num_buckets: int = 256, max_log: float = MAX_LOG) -> np.ndarray:
    """
    Approx inverse of:
        bucket = floor((log1p(dt_ns) / max_log) * num_buckets)
    We map bucket -> midpoint in log space -> expm1 -> seconds.
    """
    b = buckets.astype(np.float64)
    log_dt = ((b + 0.5) / num_buckets) * max_log
    dt_ns = np.expm1(log_dt)
    return dt_ns / 1e9

def load_id_to_event(path: str) -> dict[int, str]:
    with open(path, "r") as f:
        raw = json.load(f)
    # your file stores keys as strings
    return {int(k): v for k, v in raw.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="synthetic .npz file (event, dt, cpu)")
    ap.add_argument("--id_to_event", required=True, help="metadata_all_events/id_to_event.json")
    ap.add_argument("--out", required=True, help="output .txt (single file)")
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--max_log", type=float, default=MAX_LOG)
    ap.add_argument("--limit_windows", type=int, default=50, help="how many windows to export (avoid huge files)")
    ap.add_argument("--start_window", type=int, default=0)
    ap.add_argument("--include_bucket", action="store_true", help="also print dt_bucket")
    args = ap.parse_args()

    id2ev = load_id_to_event(args.id_to_event)

    with np.load(args.npz) as d:
        event = d["event"]  # [B,L]
        dt_b  = d["dt"]     # [B,L] bucket ids
        cpu   = d["cpu"]    # [B,L]

    B, L = event.shape
    s = max(0, int(args.start_window))
    e = min(B, s + int(args.limit_windows))
    if s >= e:
        raise SystemExit(f"Nothing to export: start_window={s}, B={B}, limit={args.limit_windows}")

    # reconstruct dt in seconds (approx)
    dt_sec = bucket_to_dt_seconds(dt_b[s:e], num_buckets=args.num_dt_buckets, max_log=args.max_log)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# Source npz: {args.npz}\n")
        f.write(f"# Windows exported: [{s}, {e}) out of B={B}, seq_len={L}\n")
        f.write(f"# dt inverse uses: max_log={args.max_log}, num_dt_buckets={args.num_dt_buckets}\n")
        f.write("# NOTE: dt_sec is an approximate inverse of bucketization.\n\n")

        for wi in range(s, e):
            local_i = wi - s
            f.write(f"=== WINDOW {wi} ===\n")
            for t in range(L):
                ev_id = int(event[wi, t])
                ev_name = id2ev.get(ev_id, f"<UNK:{ev_id}>")
                cpu_id = int(cpu[wi, t])
                dt_s = float(dt_sec[local_i, t])

                if args.include_bucket:
                    b = int(dt_b[wi, t])
                    f.write(f"{t:04d} cpu={cpu_id} dt_bucket={b:3d} dt_sec={dt_s:.9e} event={ev_name}\n")
                else:
                    f.write(f"{t:04d} cpu={cpu_id} dt_sec={dt_s:.9e} event={ev_name}\n")
            f.write("\n")

    print(f"[WRITE] {args.out} (windows {s}..{e-1})")

if __name__ == "__main__":
    main()



#Export 500 windows (100k events):
# python dataset_processing/npz_to_trace_txt.py --npz generated_traces/discrete/compress-gzip/synth_100k_step50k.npz --id_to_event metadata_all_events/id_to_event.json --out generated_traces/discrete/compress-gzip/synth_preview_500w.txt --limit_windows 500 --include_bucket

#Export all windows (will be huge: 20 million lines-ish):
#python dataset_processing/npz_to_trace_txt.py --npz generated_traces/discrete/compress-gzip/synth_100k_step50k.npz --id_to_event metadata_all_events/id_to_event.json --out generated_traces/discrete/compress-gzip/synth_full_100k.txt --include_bucket
