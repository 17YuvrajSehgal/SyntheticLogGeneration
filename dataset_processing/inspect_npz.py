import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("--n", type=int, default=3, help="num sequences to print")
    ap.add_argument("--k", type=int, default=15, help="top-k events to show")
    args = ap.parse_args()

    d = np.load(args.npz_path)
    event = d["event"]
    dt    = d["dt"]
    cpu   = d["cpu"]

    print("=== SHAPES ===")
    print("event:", event.shape, event.dtype)
    print("dt   :", dt.shape, dt.dtype)
    print("cpu  :", cpu.shape, cpu.dtype)

    B, L = event.shape

    print("\n=== SAMPLE SEQUENCES (first 20 tokens) ===")
    for i in range(min(args.n, B)):
        print(f"\n-- seq {i} --")
        print("event:", event[i, :20].tolist())
        print("dt   :", dt[i, :20].tolist())
        print("cpu  :", cpu[i, :20].tolist())

    print("\n=== BASIC STATS ===")
    print("event min/max:", int(event.min()), int(event.max()))
    print("dt    min/max:", int(dt.min()), int(dt.max()))
    print("cpu   min/max:", int(cpu.min()), int(cpu.max()))

    # distributions
    ev_counts = np.bincount(event.reshape(-1))
    top = np.argsort(ev_counts)[::-1][:args.k]
    print(f"\nTop {args.k} events:")
    for eid in top:
        if ev_counts[eid] == 0:
            break
        print(f"  event_id={eid}: {int(ev_counts[eid])}")

    cpu_counts = np.bincount(cpu.reshape(-1))
    print("\nCPU distribution:")
    for c in range(len(cpu_counts)):
        if cpu_counts[c] > 0:
            print(f"  cpu {c}: {int(cpu_counts[c])}")

    dt_flat = dt.reshape(-1).astype(np.int64)
    print("\nDT bucket stats:")
    print("  min=", int(dt_flat.min()), "max=", int(dt_flat.max()), "mean=", float(dt_flat.mean()))

if __name__ == "__main__":
    main()
