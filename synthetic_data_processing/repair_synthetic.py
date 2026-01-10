import argparse
import json
import numpy as np
import os
import glob
from tqdm import tqdm

def load_npz(path):
    with np.load(path) as data:
        return data['event'], data['dt'], data['cpu']

def load_constraints(path):
    with open(path, 'r') as f:
        return json.load(f)

def build_reference_store(real_glob, max_windows=10000):
    """
    Load a buffer of real windows to use for repair patching.
    """
    print(f"[INFO] Building reference store from {real_glob}...")
    files = sorted(glob.glob(real_glob))
    if not files:
        raise ValueError("No real files found for reference store.")
        
    ref_events = []
    ref_dts = []
    ref_cpus = []
    
    count = 0
    for f in files:
        ev, dt, cpu = load_npz(f)
        # Randomly sample if file is huge? Or just take first N?
        # Let's take first N for stability
        take = min(len(ev), max_windows - count)
        
        ref_events.append(ev[:take])
        ref_dts.append(dt[:take])
        ref_cpus.append(cpu[:take])
        
        count += take
        if count >= max_windows:
            break
            
    if count == 0:
        raise ValueError("No data found in real shards.")
        
    return (
        np.concatenate(ref_events, axis=0),
        np.concatenate(ref_dts, axis=0),
        np.concatenate(ref_cpus, axis=0)
    )

def find_nearest_real(target_event, ref_events):
    """
    Find index of real window with smallest Hamming distance to target_event sequence.
    Optimized: process in chunks if ref_events is huge.
    """
    # Simple Hamming: count non-matches
    # (ref_events != target_event) -> boolean matrix
    # sum(axis=1) -> distance
    
    # Using broadcasting: [N_ref, L] vs [L]
    diff = (ref_events != target_event)
    dists = diff.sum(axis=1)
    best_idx = np.argmin(dists)
    return best_idx, dists[best_idx]

def repair_synthetic(synth_path, constraints_path, real_glob, output_path, max_ref_windows=10000):
    print(f"[INFO] Repairing {synth_path}...")
    
    # 1. Load Data
    ev, dt, cpu = load_npz(synth_path)
    B, L = ev.shape
    
    # 2. Load Constraints
    c = load_constraints(constraints_path)
    allowed_trans = c["allowed_transitions"] # dict str->list
    dt_bounds = c["dt_constraints"] # dict str->dict
    allowed_cpus = set(c["allowed_cpus"])
    
    # Convert allowed_trans to set of tuples for fast check
    allowed_trans_set = set()
    for src, dsts in allowed_trans.items():
        for dst in dsts:
            allowed_trans_set.add((int(src), int(dst)))
            
    # 3. Load Reference Store for Patching
    ref_ev, ref_dt, ref_cpu = build_reference_store(real_glob, max_ref_windows)
    print(f"[INFO] Reference store size: {len(ref_ev)}")
    
    # 4. Iterate and Repair
    repaired_ev = ev.copy()
    repaired_dt = dt.copy()
    repaired_cpu = cpu.copy()
    
    total_violations = 0
    repaired_windows_count = 0
    
    for i in tqdm(range(B), desc="Checking windows"):
        window_ev = ev[i]
        window_dt = dt[i]
        window_cpu = cpu[i]
        
        is_bad = False
        
        # Check transitions
        # Vectorized check for python loop speed? 
        # python set lookup is fast enough for 200 items
        for t in range(L - 1):
            pair = (window_ev[t], window_ev[t+1])
            if pair not in allowed_trans_set:
                is_bad = True
                break
        
        if not is_bad:
            # Check CPU
            for t in range(L):
                if window_cpu[t] not in allowed_cpus:
                    is_bad = True
                    break
                    
        if not is_bad:
            # Check DT
            for t in range(L):
                e_id = str(window_ev[t])
                # If event not in constraints, assume 0-255 allowed (default)
                if e_id in dt_bounds:
                    # check range
                    # For strictness: check if in 'allowed_set'
                    # But 'allowed_set' might be sparse. Let's strictly use allowed_set if available
                    allowed_set = set(dt_bounds[e_id]["allowed_set"])
                    if not allowed_set: 
                        continue # no constraints?
                    
                    if window_dt[t] not in allowed_set:
                         is_bad = True
                         break
        
        if is_bad:
            total_violations += 1
            
            # --- REPAIR STRATEGY ---
            # "Tracyn": Replace with nearest real window
            # Ideally we might only replace the Bad Segment, but window-replacement is cleaner guarantee
            # and easier to implement for V1.
            
            best_idx, dist = find_nearest_real(window_ev, ref_ev)
            
            # Replace
            repaired_ev[i] = ref_ev[best_idx]
            repaired_dt[i] = ref_dt[best_idx]
            repaired_cpu[i] = ref_cpu[best_idx]
            repaired_windows_count += 1
            
    print(f"[RESULT] Total windows: {B}")
    print(f"[RESULT] Violating windows: {total_violations} ({total_violations/B*100:.2f}%)")
    print(f"[RESULT] Repaired windows: {repaired_windows_count}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, event=repaired_ev, dt=repaired_dt, cpu=repaired_cpu)
    print(f"[INFO] Saved repaired traces to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth_npz", required=True)
    parser.add_argument("--constraints_json", required=True)
    parser.add_argument("--real_glob", required=True, help="Glob for real data to build reference store")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--max_ref", type=int, default=10000)
    
    args = parser.parse_args()
    
    repair_synthetic(args.synth_npz, args.constraints_json, args.real_glob, args.output_npz, args.max_ref)
