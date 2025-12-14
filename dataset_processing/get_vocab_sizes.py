import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()

    d = np.load(args.shard)
    max_event = int(d["event"].max())
    max_cpu = int(d["cpu"].max())
    max_dt = int(d["dt"].max())

    print("max_event_id:", max_event)
    print("num_events  :", max_event + 1)
    print("max_cpu     :", max_cpu)
    print("max_dt_bucket:", max_dt)

if __name__ == "__main__":
    main()
