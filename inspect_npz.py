import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to generated .npz file")
    ap.add_argument("--n", type=int, default=3, help="Number of sequences to print")
    ap.add_argument("--k", type=int, default=15, help="Top-K events to show")
    args = ap.parse_args()

    d = np.load(args.npz)

    event = d["event"]
    dt    = d["dt"]
    cpu   = d["cpu"]

    print("=== FILE INFO ===")
    print("Path:", args.npz)
    print("Keys:", list(d.keys()))
    print()

    print("=== SHAPES & DTYPES ===")
    print("event:", event.shape, event.dtype)
    print("dt   :", dt.shape, dt.dtype)
    print("cpu  :", cpu.shape, cpu.dtype)
    print()

    B, L = event.shape

    print("=== VALUE RANGES ===")
    print("event min/max:", int(event.min()), int(event.max()))
    print("dt    min/max:", int(dt.min()), int(dt.max()))
    print("cpu   min/max:", int(cpu.min()), int(cpu.max()))
    print()

    print("=== SAMPLE SEQUENCES (first 20 tokens) ===")
    for i in range(min(args.n, B)):
        print(f"\n-- seq {i} --")
        print("event:", event[i, :20].tolist())
        print("dt   :", dt[i, :20].tolist())
        print("cpu  :", cpu[i, :20].tolist())

    print("\n=== DISTRIBUTIONS ===")

    ev_counts = np.bincount(event.reshape(-1))
    top = np.argsort(ev_counts)[::-1][:args.k]
    print(f"Top {args.k} events:")
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
    print("\nDT statistics:")
    print("  mean =", float(dt_flat.mean()))
    print("  median =", float(np.median(dt_flat)))
    print("  std =", float(dt_flat.std()))
    print("  p1 / p5 / p95 / p99 =",
          np.percentile(dt_flat, [1, 5, 95, 99]).tolist())

    print("\n=== SANITY CHECKS ===")
    if cpu.max() > 3:
        print("[FAIL] CPU values exceed expected range (0–3)")
    else:
        print("[OK] CPU values in range")

    if dt.min() < 0:
        print("[FAIL] Negative dt detected")
    else:
        print("[OK] dt non-negative")

    if event.min() < 0:
        print("[FAIL] Negative event IDs detected")
    else:
        print("[OK] event IDs non-negative")

    d.close()

if __name__ == "__main__":
    main()