import argparse
import json
import numpy as np
import os
import sys

def load_constraints(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_traces(path):
    with np.load(path) as data:
        # Support both 'event' and 'events' keys just in case, but standard is 'event'
        if 'event' in data:
            return data['event'], data['dt'], data['cpu']
        else:
            raise KeyError(f"File {path} missing 'event' key")

def validate_traces(trace_path, constraints_path, output_path):
    print(f"[INFO] Validating {trace_path} against {constraints_path}...")
    
    constraints = load_constraints(constraints_path)
    events, dts, cpus = load_traces(trace_path)
    
    # Unpack constraints
    allowed_trans_adj = {int(k): set(v) for k, v in constraints["allowed_transitions"].items()}
    dt_constraints = {int(k): v for k, v in constraints["dt_constraints"].items()}
    allowed_cpus_global = set(constraints["allowed_cpus"])
    
    # New: Event-CPU constraints
    # Handle backward compatibility if key missing
    if "event_cpu_constraints" in constraints:
        event_cpu_map = {int(k): set(v) for k, v in constraints["event_cpu_constraints"].items()}
    else:
        event_cpu_map = None

    num_samples = events.shape[0]
    seq_len = events.shape[1]
    total_events = num_samples * seq_len
    
    # Metrics
    invalid_transitions = 0
    invalid_dts = 0
    invalid_cpus_global = 0
    invalid_cpus_local = 0
    
    # 1. Transition Validity
    # Vectorized approach is tricky with adjacency list. 
    # Let's use Python loop for logic clarity, or optimize if slow.
    # For reporting "Hard Guarantees", exact count matters.
    
    # check e_t -> e_{t+1}
    for i in range(num_samples):
        row_ev = events[i]
        row_dt = dts[i]
        row_cpu = cpus[i]
        
        for t in range(seq_len - 1):
            curr_e = int(row_ev[t])
            next_e = int(row_ev[t+1])
            
            # Transition
            if curr_e not in allowed_trans_adj:
                # Unknown event?
                invalid_transitions += 1
            elif next_e not in allowed_trans_adj[curr_e]:
                invalid_transitions += 1
                
        # 2. Timing Validity & 3. CPU Validity
        for t in range(seq_len):
            curr_e = int(row_ev[t])
            curr_dt = row_dt[t]
            curr_cpu = int(row_cpu[t])
            
            # DT
            if curr_e in dt_constraints:
                c = dt_constraints[curr_e]
                # Check min/max
                # Note: dt in npz is log(1+dt) usually, or raw? 
                # README says: "Log-normalized: log(1 + Delta t)"
                # Constraints were learned from the SAME format (npz). So we compare directly.
                if curr_dt < c["min"] or curr_dt > c["max"]:
                    invalid_dts += 1
            else:
                # Unknown event?
                invalid_dts += 1
                
            # CPU Global
            if curr_cpu not in allowed_cpus_global:
                invalid_cpus_global += 1
                
            # CPU Local
            if event_cpu_map and curr_e in event_cpu_map:
                if curr_cpu not in event_cpu_map[curr_e]:
                    invalid_cpus_local += 1

    # Report
    # Transitions: Total interactions = N * (L-1)
    total_trans = num_samples * (seq_len - 1)
    
    validity_trans = 100.0 * (1.0 - invalid_transitions / total_trans)
    validity_dt = 100.0 * (1.0 - invalid_dts / total_events)
    validity_cpu_global = 100.0 * (1.0 - invalid_cpus_global / total_events)
    validity_cpu_local = 100.0 * (1.0 - invalid_cpus_local / total_events) if event_cpu_map else 100.0
    
    report = {
        "validity_score": {
            "transitions": validity_trans,
            "timing": validity_dt,
            "cpu_global": validity_cpu_global,
            "cpu_local": validity_cpu_local
        },
        "details": {
            "total_samples": int(num_samples),
            "seq_len": int(seq_len),
            "total_events": int(total_events),
            "invalid_transitions": int(invalid_transitions),
            "invalid_dts": int(invalid_dts),
            "invalid_cpu_local": int(invalid_cpus_local)
        }
    }
    
    print(json.dumps(report, indent=2))
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            print(f"[INFO] Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="Path to generated .npz")
    parser.add_argument("--constraints", required=True, help="Path to constraints.json")
    parser.add_argument("--output", default="validity_report.json")
    
    args = parser.parse_args()
    
    validate_traces(args.trace, args.constraints, args.output)
