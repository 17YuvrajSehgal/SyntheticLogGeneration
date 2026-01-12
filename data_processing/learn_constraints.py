import argparse
import glob
import json
import numpy as np
import os
import multiprocessing
from tqdm import tqdm
from functools import partial

def load_npz(path):
    with np.load(path) as data:
        return data['event'], data['dt'], data['cpu'], data['tid'], data['fd'], data['comm'], data['ret']

def process_shard(path):
    """
    Process a single NPZ shard and return partial constraints.
    """
    try:
        ev, dt, cpu, tid, fd, comm, ret = load_npz(path)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None

    # Partial results
    partial_transitions = set()
    partial_dt_stats = {} # e_id -> list of dts
    partial_cpu_map = {}  # e_id -> set of cpus
    partial_tid_map = {}
    partial_fd_map = {}
    partial_comm_map = {}
    partial_ret_map = {}
    partial_counts = {}
    partial_allowed_cpus = set()

    # Flatten
    ev_flat = ev.reshape(-1)
    dt_flat = dt.reshape(-1)
    cpu_flat = cpu.reshape(-1)
    tid_flat = tid.reshape(-1)
    fd_flat = fd.reshape(-1)
    comm_flat = comm.reshape(-1)
    ret_flat = ret.reshape(-1)
    
    # 1. Transitions
    e_curr = ev[:, :-1]
    e_next = ev[:, 1:]
    transitions = np.stack([e_curr.reshape(-1), e_next.reshape(-1)], axis=1)
    unique_trans = np.unique(transitions, axis=0)
    for t in unique_trans:
        partial_transitions.add(tuple(t.tolist()))

    # 2. Per-event Stats
    unique_events = np.unique(ev_flat)
    for e_id in unique_events:
        e_id = int(e_id)
        mask = (ev_flat == e_id)
        
        # DT (store observed values, prevent explosion by using unique locally)
        dts = np.unique(dt_flat[mask]).tolist()
        partial_dt_stats[e_id] = dts
        
        # CPU
        cpus = np.unique(cpu_flat[mask]).tolist()
        if e_id not in partial_cpu_map: partial_cpu_map[e_id] = set()
        partial_cpu_map[e_id].update(cpus)
        partial_allowed_cpus.update(cpus)
        
        # TID
        tids = np.unique(tid_flat[mask]).tolist()
        if e_id not in partial_tid_map: partial_tid_map[e_id] = set()
        partial_tid_map[e_id].update(tids)

        # FD
        fds = np.unique(fd_flat[mask]).tolist()
        if e_id not in partial_fd_map: partial_fd_map[e_id] = set()
        partial_fd_map[e_id].update(fds)
        
        # COMM
        comms = np.unique(comm_flat[mask]).tolist()
        if e_id not in partial_comm_map: partial_comm_map[e_id] = set()
        partial_comm_map[e_id].update(comms)
        
        # RET
        rets = np.unique(ret_flat[mask]).tolist()
        if e_id not in partial_ret_map: partial_ret_map[e_id] = set()
        partial_ret_map[e_id].update(rets)
        
        # Counts
        partial_counts[e_id] = int(np.sum(mask))
        
    return (partial_transitions, partial_dt_stats, partial_allowed_cpus, partial_cpu_map, 
            partial_tid_map, partial_fd_map, partial_comm_map, partial_ret_map, partial_counts)

def learn_constraints(real_glob, num_events, num_cpus, output_path, workers=None):
    print(f"[INFO] Learning constraints from {real_glob}...")
    
    files = glob.glob(real_glob, recursive=True)
    if "**" not in real_glob and "*.npz" in real_glob:
        alt_glob = real_glob.replace("*.npz", "**/*.npz")
        files.extend(glob.glob(alt_glob, recursive=True))
    files = sorted(list(set(files)))
    
    if not files:
        print(f"[ERROR] No files found matching {real_glob}")
        return

    # Multiprocessing
    if workers is None:
        workers = min(multiprocessing.cpu_count(), 32) # Cap at 32 or Available
    
    print(f"[INFO] Processing {len(files)} shards with {workers} workers...")
    
    # Global Aggregators
    all_transitions = set()
    all_dt_stats = {e: set() for e in range(num_events)}
    all_allowed_cpus = set()
    
    all_cpu_map = {e: set() for e in range(num_events)}
    all_tid_map = {e: set() for e in range(num_events)}
    all_fd_map = {e: set() for e in range(num_events)}
    all_comm_map = {e: set() for e in range(num_events)}
    all_ret_map = {e: set() for e in range(num_events)}
    
    all_counts = {e: 0 for e in range(num_events)}

    with multiprocessing.Pool(workers) as pool:
        # Use imap_unordered for better responsiveness with tqdm
        results = list(tqdm(pool.imap_unordered(process_shard, files), total=len(files), desc="Mining constraints"))
        
    print("[INFO] Aggregating results...")
    for res in results:
        if res is None: continue
        (p_trans, p_dt, p_allowed_cpus, p_cpu, p_tid, p_fd, p_comm, p_ret, p_cnt) = res
        
        all_transitions.update(p_trans)
        all_allowed_cpus.update(p_allowed_cpus)
        
        for e, count in p_cnt.items():
            all_counts[e] += count
            
        for e, dts in p_dt.items():
            all_dt_stats[e].update(dts)
            
        for e, vals in p_cpu.items(): all_cpu_map[e].update(vals)
        for e, vals in p_tid.items(): all_tid_map[e].update(vals)
        for e, vals in p_fd.items(): all_fd_map[e].update(vals)
        for e, vals in p_comm.items(): all_comm_map[e].update(vals)
        for e, vals in p_ret.items(): all_ret_map[e].update(vals)

    # Post-process
    # DT Stats
    final_dt_constraints = {}
    for e_id, dts in all_dt_stats.items():
        if dts:
            unique_dts = sorted(list(dts))
            final_dt_constraints[e_id] = {
                "min": float(min(unique_dts)),
                "max": float(max(unique_dts)),
                "allowed_set": [float(x) for x in unique_dts]
            }
        else:
            final_dt_constraints[e_id] = {"min": 0.0, "max": 0.0, "allowed_set": []}

    # Transitions
    adj_list = {e: [] for e in range(num_events)}
    for (src, dst) in all_transitions:
        adj_list[int(src)].append(int(dst))
    for e in adj_list: adj_list[e].sort()

    # Maps
    final_cpu_constraints = {e: sorted(list(vals)) for e, vals in all_cpu_map.items()}
    final_tid_constraints = {e: sorted(list(vals)) for e, vals in all_tid_map.items()}
    final_fd_constraints = {e: sorted(list(vals)) for e, vals in all_fd_map.items()}
    final_comm_constraints = {e: sorted(list(vals)) for e, vals in all_comm_map.items()}
    final_ret_constraints = {e: sorted(list(vals)) for e, vals in all_ret_map.items()}

    # Probabilities
    total_events = sum(all_counts.values())
    event_probs = {e: count / total_events if total_events > 0 else 0.0 for e, count in all_counts.items()}

    constraints = {
        "allowed_transitions": adj_list,
        "dt_constraints": final_dt_constraints,
        "allowed_cpus": sorted(list(all_allowed_cpus)),
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
    if output_dir: os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w') as f:
        json.dump(constraints, f, indent=2)
        
    print(f"[INFO] Done. Found {len(all_transitions)} unique transitions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_glob", required=True, help="Glob pattern for real training shards (recursive)")
    parser.add_argument("--output", required=True, help="Path to save constraints.json")
    parser.add_argument("--num_events", type=int, default=384, help="Vocabulary size of Event IDs")
    parser.add_argument("--num_cpus", type=int, default=4, help="Number of CPUs (for metadata)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: all cpus)")
    
    args = parser.parse_args()
    
    learn_constraints(args.real_glob, args.num_events, args.num_cpus, args.output, args.workers)