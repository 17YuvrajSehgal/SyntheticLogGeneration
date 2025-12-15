import argparse, glob, numpy as np

def kl(p, q, eps=1e-12):
    p = p.astype(np.float64); q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def hist(arr, size):
    arr = arr.reshape(-1).astype(np.int64)
    return np.bincount(arr, minlength=size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_glob", required=True, help='e.g., "window_shards/compress-gzip/train/*.npz"')
    ap.add_argument("--synth", required=True)
    ap.add_argument("--num_events", type=int, default=380)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)
    ap.add_argument("--max_shards", type=int, default=50)
    args = ap.parse_args()

    real_paths = sorted(glob.glob(args.real_glob))
    assert real_paths, f"No real shards match {args.real_glob}"
    real_paths = real_paths[:args.max_shards]

    R_event = np.zeros(args.num_events, dtype=np.int64)
    R_dt    = np.zeros(args.num_dt_buckets, dtype=np.int64)
    R_cpu   = np.zeros(args.num_cpus, dtype=np.int64)

    for p in real_paths:
        d = np.load(p)
        R_event += hist(d["event"], args.num_events)
        R_dt    += hist(d["dt"], args.num_dt_buckets)
        R_cpu   += hist(d["cpu"], args.num_cpus)
        d.close()

    S = np.load(args.synth)
    S_event = hist(S["event"], args.num_events)
    S_dt    = hist(S["dt"], args.num_dt_buckets)
    S_cpu   = hist(S["cpu"], args.num_cpus)
    S.close()

    print("Real shards used:", len(real_paths))
    print("=== KL Divergence (RealAgg || Synth) ===")
    print("Event KL:", kl(R_event, S_event))
    print("DT KL   :", kl(R_dt, S_dt))
    print("CPU KL  :", kl(R_cpu, S_cpu))

if __name__ == "__main__":
    main()
