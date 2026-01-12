import argparse
import json
import numpy as np
import os
import random
from tqdm import tqdm

def load_constraints(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_traces(path):
    with np.load(path) as data:
        # Load all keys to preserve them
        return dict(data)

def repair_traces(trace_path, constraints_path, output_path):
    print(f"[INFO] Repairing {trace_path} using {constraints_path}...")
    
    constraints = load_constraints(constraints_path)
    data = load_traces(trace_path)
    
    events = data['event'].copy() # [N, L]
    dt = data['dt'].copy() # [N, L]
    # We might need to modify other channels if we change events (e.g. cpu constraint)
    # For now, let's focus on Event Sequence and CPU consistency.
    cpus = data['cpu'].copy()
    
    allowed_trans_adj = {int(k): list(v) for k, v in constraints["allowed_transitions"].items()}
    # event_probs = {int(k): v for k, v in constraints.get("event_probs", {}).items()}
    
    # DT Constraints
    dt_constraints = {int(k): v for k, v in constraints.get("dt_constraints", {}).items()}

    # Local CPU map
    if "event_cpu_constraints" in constraints:
        event_cpu_map = {int(k): list(v) for k, v in constraints["event_cpu_constraints"].items()}
    else:
        event_cpu_map = None

    num_samples, seq_len = events.shape
    repairs_made = 0
    total_checks = 0
    
    # Strategy: Greedy Forward Repair
    # If e_t -> e_{t+1} is invalid:
    #   Resample e_{t+1} from Allowed(e_t).
    #   Prefer events that are likely? Or random?
    #   For "Hard Guarantees", just picking ANY valid next event satisfies the immediate transition.
    
    for i in tqdm(range(num_samples), desc="Repairing traces"):
        # 1. Event Transition Repair
        for t in range(seq_len - 1):
            curr_e = int(events[i, t])
            next_e = int(events[i, t+1])
            
            # Check validity
            if curr_e in allowed_trans_adj:
                allowed_next = allowed_trans_adj[curr_e]
                if next_e not in allowed_next:
                    # Invalid! Repair.
                    if allowed_next:
                        # Pick a valid replacement
                        # TODO: Use probability distribution if available?
                        # For now: Random valid
                        fixed_next = random.choice(allowed_next)
                        events[i, t+1] = fixed_next
                        
                        # Fix DT for the new event!
                        # The old dt was for the old event. New event might need different timing.
                        if fixed_next in dt_constraints:
                            c = dt_constraints[fixed_next]
                            # Sample from allowed set if available, else clamp
                            if c.get("allowed_set"):
                                fixed_dt = random.choice(c["allowed_set"])
                            else:
                                fixed_dt = c["min"] # Fallback
                            dt[i, t+1] = fixed_dt
                            
                        repairs_made += 1
                    else:
                        # Dead end event (e.g. crash/exit)? 
                        # If the constraint says NO transitions allowed, but we have a next event,
                        # we are in trouble. Maybe terminate trace? (Pad with 0)
                        pass
            
            total_checks += 1
            
        # 2. CPU Consistency Repair
        if event_cpu_map:
            for t in range(seq_len):
                curr_e = int(events[i, t])
                curr_cpu = int(cpus[i, t])
                
                if curr_e in event_cpu_map:
                    allowed_cpus_for_event = event_cpu_map[curr_e]
                    if curr_cpu not in allowed_cpus_for_event:
                        # Event on wrong CPU.
                        # Fix: Move to allowed CPU.
                        if allowed_cpus_for_event:
                            fixed_cpu = random.choice(allowed_cpus_for_event)
                            cpus[i, t] = fixed_cpu
                            # Note: This might break TID consistency if we tracked that carefully.
                        
    # Update data dict
    data['event'] = events
    data['cpu'] = cpus
    data['dt'] = dt
    
    print(f"[INFO] Repairs made: {repairs_made} / {total_checks} transitions.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **data)
    print(f"[INFO] Repaired trace saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="Path to input .npz")
    parser.add_argument("--constraints", required=True, help="Path to constraints.json")
    parser.add_argument("--output", required=True, help="Path to output repaired .npz")
    
    args = parser.parse_args()
    
    repair_traces(args.trace, args.constraints, args.output)
