import argparse, numpy as np

def kl(p, q, eps=1e-12):
    p = p.astype(np.float64); q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def hist(arr, size=None):
    arr = arr.reshape(-1).astype(np.int64)
    if size is None:
        size = int(arr.max()) + 1
    h = np.bincount(arr, minlength=size)
    return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="path to a real shard .npz (window_shards/.../train/*.npz)")
    ap.add_argument("--synth", required=True, help="path to synthetic .npz")
    ap.add_argument("--num_events", type=int, default=380)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)
    args = ap.parse_args()

    R = np.load(args.real)
    S = np.load(args.synth)

    r_event, r_dt, r_cpu = R["event"], R["dt"], R["cpu"]
    s_event, s_dt, s_cpu = S["event"], S["dt"], S["cpu"]

    re = hist(r_event, args.num_events)
    se = hist(s_event, args.num_events)

    rd = hist(r_dt, args.num_dt_buckets)
    sd = hist(s_dt, args.num_dt_buckets)

    rc = hist(r_cpu, args.num_cpus)
    sc = hist(s_cpu, args.num_cpus)

    print("=== KL Divergence (Real || Synth) ===")
    print("Event KL:", kl(re, se))
    print("DT KL   :", kl(rd, sd))
    print("CPU KL  :", kl(rc, sc))

    print("\n=== Quick ranges ===")
    print("Real  event min/max:", int(r_event.min()), int(r_event.max()))
    print("Synth event min/max:", int(s_event.min()), int(s_event.max()))
    print("Real  dt    min/max:", int(r_dt.min()), int(r_dt.max()))
    print("Synth dt    min/max:", int(s_dt.min()), int(s_dt.max()))
    print("Real  cpu   min/max:", int(r_cpu.min()), int(r_cpu.max()))
    print("Synth cpu   min/max:", int(s_cpu.min()), int(s_cpu.max()))

if __name__ == "__main__":
    main()
