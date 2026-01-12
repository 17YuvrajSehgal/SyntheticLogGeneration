import argparse
import glob
import json
import numpy as np
import os
from tqdm import tqdm

def load_npz(path):
    with np.load(path) as data:
        return data['event'], data['dt'], data['cpu'], data['tid'], data['fd'], data['comm'], data['ret']

def learn_constraints(real_glob, num_events, num_cpus, output_path):
    print(f"[INFO] Learning constraints from {real_glob}...")
    
    # Initialize structures
    allowed_transitions = set()  # (e_t, e_{t+1})
    event_dt_stats = {e: [] for e in range(num_events)} # store observed dt buckets for each event
    allowed_cpus = set()
    
    # New: Per-event constraints
    event_cpu_map = {e: set() for e in range(num_events)}
    event_tid_map = {e: set() for e in range(num_events)}
    event_fd_map = {e: set() for e in range(num_events)}
    event_comm_map = {e: set() for e in range(num_events)}
    event_ret_map = {e: set() for e in range(num_events)}
    
    event_counts = {e: 0 for e in range(num_events)}
    
    files = glob.glob(real_glob, recursive=True)
    
    # Auto-enable recursion if user didn't specify ** but possibly meant it
    if "**" not in real_glob and "*.npz" in real_glob:
        alt_glob = real_glob.replace("*.npz", "**/*.npz")
        files_alt = glob.glob(alt_glob, recursive=True)
        files.extend(files_alt)
        
    # Deduplicate and sort
    files = sorted(list(set(files)))
    
    if not files:
        print(f"[ERROR] No files found matching {real_glob} (recursive search attempted)")
        return

    for f in tqdm(files, desc="Processing shards"):
        ev, dt, cpu, tid, fd, comm, ret = load_npz(f)
        
        # Flatten for simpler processing
        ev_flat = ev.reshape(-1)
        dt_flat = dt.reshape(-1)
        cpu_flat = cpu.reshape(-1)
        tid_flat = tid.reshape(-1)
        fd_flat = fd.reshape(-1)
        comm_flat = comm.reshape(-1)
        ret_flat = ret.reshape(-1)
        
        # 1. Allowed transitions (Bigrams)
        # Vectorized bigram extraction
        e_curr = ev[:, :-1]
        e_next = ev[:, 1:]
        
        transitions = np.stack([e_curr.reshape(-1), e_next.reshape(-1)], axis=1)
        unique_trans = np.unique(transitions, axis=0)
        for t in unique_trans:
            allowed_transitions.add(tuple(t.tolist()))

        # 2. Event-conditioned constraints
        unique_events_in_shard = np.unique(ev_flat)
        
        for e_id in unique_events_in_shard:
            mask = (ev_flat == e_id)
            
            # DT
            observed_dts = np.unique(dt_flat[mask])
            event_dt_stats[int(e_id)].extend(observed_dts.tolist())
            
            # CPU
            observed_cpus_local = np.unique(cpu_flat[mask])
            for c in observed_cpus_local:
                event_cpu_map[int(e_id)].add(int(c))
                allowed_cpus.add(int(c))
            
            # TID
            for v in np.unique(tid_flat[mask]): event_tid_map[int(e_id)].add(int(v))
            
            # FD
            for v in np.unique(fd_flat[mask]): event_fd_map[int(e_id)].add(int(v))
            
            # COMM
            for v in np.unique(comm_flat[mask]): event_comm_map[int(e_id)].add(int(v))
            
            # RET
            for v in np.unique(ret_flat[mask]): event_ret_map[int(e_id)].add(int(v))
                
            # Counts
            event_counts[int(e_id)] += int(np.sum(mask))

    # Post-process
    
    # Consolidate dt stats: instead of big list, just keep the Set of allowed buckets
    final_dt_constraints = {}
    for e_id, dts in event_dt_stats.items():
        if dts:
            unique_dts = sorted(list(set(dts)))
            final_dt_constraints[e_id] = {
                "min": float(min(unique_dts)),
                "max": float(max(unique_dts)),
                "allowed_set": [float(x) for x in unique_dts] # List for JSON
            }
        else:
            # Event never observed?
            final_dt_constraints[e_id] = {
                "min": 0.0,
                "max": 0.0,
                "allowed_set": []
            }

    # Format allowed transitions as "e_from": [list of allowed e_to]
    adj_list = {e: [] for e in range(num_events)}
    for (src, dst) in allowed_transitions:
        adj_list[int(src)].append(int(dst))
        
    for e in adj_list:
        adj_list[e].sort()

    # Format CPU maps and others
    final_cpu_constraints = {e: sorted(list(vals)) for e, vals in event_cpu_map.items()}
    final_tid_constraints = {e: sorted(list(vals)) for e, vals in event_tid_map.items()}
    final_fd_constraints = {e: sorted(list(vals)) for e, vals in event_fd_map.items()}
    final_comm_constraints = {e: sorted(list(vals)) for e, vals in event_comm_map.items()}
    final_ret_constraints = {e: sorted(list(vals)) for e, vals in event_ret_map.items()}
    
    # Calculate Probabilities
    total_events = sum(event_counts.values())
    event_probs = {e: count / total_events if total_events > 0 else 0.0 for e, count in event_counts.items()}

    constraints = {
        "allowed_transitions": adj_list,
        "dt_constraints": final_dt_constraints,
        "allowed_cpus": sorted(list(allowed_cpus)),
        "event_cpu_constraints": final_cpu_constraints,
        "event_tid_constraints": final_tid_constraints,
        "event_fd_constraints": final_fd_constraints,
        "event_comm_constraints": final_comm_constraints,
        "event_ret_constraints": final_ret_constraints,
        "event_probs": event_probs,
        "metadata": {
            "num_events": num_events,
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
    parser.add_argument("--real_glob", required=True, help="Glob pattern for real training shards (recursive)")
    parser.add_argument("--output", required=True, help="Path to save constraints.json")
    parser.add_argument("--num_events", type=int, default=384, help="Vocabulary size of Event IDs")
    parser.add_argument("--num_cpus", type=int, default=4, help="Number of CPUs (for metadata)")
    # limit the number of shards to process for speed?
    
    args = parser.parse_args()
    
    learn_constraints(args.real_glob, args.num_events, args.num_cpus, args.output)

# python data_processing/learn_constraints.py --real_glob "dataset/window_shards/windowed_npz_256/**/*.npz" --output dataset/constraints_universal.json --num_events 384 --num_cpus 4