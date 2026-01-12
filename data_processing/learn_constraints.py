import argparse
import glob
import json
import numpy as np
import os
from tqdm import tqdm

def load_npz(path):
    with np.load(path) as data:
        return data['event'], data['dt'], data['cpu']

def learn_constraints(real_glob, num_events, num_dt_buckets, num_cpus, output_path):
    print(f"[INFO] Learning constraints from {real_glob}...")
    
    # Initialize structures
    allowed_transitions = set()  # (e_t, e_{t+1})
    event_dt_stats = {e: [] for e in range(num_events)} # store observed dt buckets for each event
    allowed_cpus = set()
    
    files = sorted(glob.glob(real_glob))
    if not files:
        print(f"[ERROR] No files found matching {real_glob}")
        return

    for f in tqdm(files, desc="Processing shards"):
        ev, dt, cpu = load_npz(f)
        
        # Flatten for simpler processing
        ev_flat = ev.reshape(-1)
        dt_flat = dt.reshape(-1)
        cpu_flat = cpu.reshape(-1)
        
        # 1. Allowed transitions (Bigrams)
        # Iterate over windows to respect boundaries? 
        # Actually, transitions happen within windows. Between windows is undefined in this dataset format (independent shuffling)
        # So we process row by row.
        
        # Vectorized bigram extraction
        e_curr = ev[:, :-1]
        e_next = ev[:, 1:]
        
        # We can zip them. To be fast, we can use unique rows
        # stack: (N*(L-1), 2)
        transitions = np.stack([e_curr.reshape(-1), e_next.reshape(-1)], axis=1)
        unique_trans = np.unique(transitions, axis=0)
        
        for t in unique_trans:
            allowed_transitions.add(tuple(t.tolist()))

        # 2. Event-conditioned DT stats
        # For memory efficiency, we won't store ALL dt values. We can store min/max and maybe a histogram if we want percentiles later.
        # But for "hard guarantees", min/max or "observed set" is best.
        # Let's simple store the set of observed dt buckets for each event.
        
        # Group dt by event
        # This can be slow in pure python. Let's use numpy.
        for e_id in np.unique(ev_flat):
            mask = (ev_flat == e_id)
            observed_dts = np.unique(dt_flat[mask])
            event_dt_stats[int(e_id)].extend(observed_dts.tolist())
            
        # 3. Allowed CPUs
        unique_cpus = np.unique(cpu_flat)
        for c in unique_cpus:
            allowed_cpus.add(int(c))
            
    # Post-process
    
    # Consolidate dt stats: instead of big list, just keep the Set of allowed buckets
    final_dt_constraints = {}
    for e_id, dts in event_dt_stats.items():
        if dts:
            unique_dts = sorted(list(set(dts)))
            final_dt_constraints[e_id] = {
                "min": min(unique_dts),
                "max": max(unique_dts),
                "allowed_set": unique_dts # List for JSON
            }
        else:
            # Event never observed?
            final_dt_constraints[e_id] = {
                "min": 0,
                "max": num_dt_buckets - 1,
                "allowed_set": []
            }

    # Format allowed transitions as "e_from": [list of allowed e_to]
    adj_list = {e: [] for e in range(num_events)}
    for (src, dst) in allowed_transitions:
        adj_list[int(src)].append(int(dst))
        
    for e in adj_list:
        adj_list[e].sort()

    constraints = {
        "allowed_transitions": adj_list,
        "dt_constraints": final_dt_constraints,
        "allowed_cpus": sorted(list(allowed_cpus)),
        "metadata": {
            "num_events": num_events,
            "num_dt_buckets": num_dt_buckets,
            "num_cpus": num_cpus
        }
    }
    
    print(f"[INFO] Saving constraints to {output_path}...")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w') as f:
        json.dump(constraints, f, indent=2)
        
    print(f"[INFO] Done. Found {len(allowed_transitions)} unique transitions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_glob", required=True, help="Glob pattern for real training shards")
    parser.add_argument("--output", required=True, help="Path to save constraints.json")
    parser.add_argument("--num_events", type=int, default=384)
    parser.add_argument("--num_dt_buckets", type=int, default=256)
    parser.add_argument("--num_cpus", type=int, default=4)
    # limit number of shards to process for speed?
    
    args = parser.parse_args()
    
    learn_constraints(args.real_glob, args.num_events, args.num_dt_buckets, args.num_cpus, args.output)
