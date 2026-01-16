"""
Diagnostic script to analyze synthetic vs real data quality.

Usage:
    python experiments_downstream/diagnose_data_quality.py \
        --real-data experiments_downstream/data/real_train.npz \
        --synthetic-data experiments_downstream/data/synthetic_repaired_1024.npz
"""

import argparse
import numpy as np
from collections import Counter
import json


def analyze_event_distribution(data, name):
    """Analyze event distribution."""
    events = data['event'].flatten()
    
    print(f"\n{'='*60}")
    print(f"{name} - Event Distribution")
    print(f"{'='*60}")
    
    print(f"Shape: {data['event'].shape}")
    print(f"Total events: {len(events)}")
    print(f"Unique events: {len(np.unique(events))}")
    print(f"Event range: [{events.min()}, {events.max()}]")
    
    # Top 10 most common events
    counter = Counter(events)
    print(f"\nTop 10 most common events:")
    for event, count in counter.most_common(10):
        pct = 100 * count / len(events)
        print(f"  Event {event:3d}: {count:8d} ({pct:5.2f}%)")
    
    # Check for concentration
    top10_count = sum(count for _, count in counter.most_common(10))
    top10_pct = 100 * top10_count / len(events)
    print(f"\nTop 10 events cover: {top10_pct:.2f}% of all events")
    
    return counter


def analyze_transitions(data, name):
    """Analyze event transitions."""
    events = data['event']
    
    print(f"\n{'='*60}")
    print(f"{name} - Transition Analysis")
    print(f"{'='*60}")
    
    transitions = []
    for trace in events:
        for i in range(len(trace) - 1):
            transitions.append((trace[i], trace[i+1]))
    
    trans_counter = Counter(transitions)
    print(f"Total transitions: {len(transitions)}")
    print(f"Unique transitions: {len(trans_counter)}")
    
    print(f"\nTop 10 most common transitions:")
    for (src, dst), count in trans_counter.most_common(10):
        pct = 100 * count / len(transitions)
        print(f"  {src:3d} -> {dst:3d}: {count:8d} ({pct:5.2f}%)")
    
    return trans_counter


def compare_distributions(real_counter, synth_counter, name="events"):
    """Compare two distributions."""
    print(f"\n{'='*60}")
    print(f"Distribution Comparison - {name}")
    print(f"{'='*60}")
    
    real_items = set(real_counter.keys())
    synth_items = set(synth_counter.keys())
    
    overlap = real_items & synth_items
    only_real = real_items - synth_items
    only_synth = synth_items - real_items
    
    print(f"Real unique {name}: {len(real_items)}")
    print(f"Synthetic unique {name}: {len(synth_items)}")
    print(f"Overlap: {len(overlap)} ({100*len(overlap)/len(real_items):.1f}% of real)")
    print(f"Only in real: {len(only_real)}")
    print(f"Only in synthetic: {len(only_synth)}")
    
    if only_real:
        print(f"\nSample {name} missing in synthetic (first 20):")
        for item in list(only_real)[:20]:
            count = real_counter[item]
            print(f"  {item}: {count} occurrences in real data")
    
    # KL divergence (simplified)
    if overlap:
        print(f"\nDistribution similarity (for overlapping {name}):")
        
        # Normalize
        real_total = sum(real_counter[k] for k in overlap)
        synth_total = sum(synth_counter[k] for k in overlap)
        
        kl_div = 0
        for item in overlap:
            p = real_counter[item] / real_total
            q = synth_counter[item] / synth_total
            if q > 0:
                kl_div += p * np.log(p / q)
        
        print(f"  KL divergence (Real || Synth): {kl_div:.4f}")
        print(f"  (Lower is better, 0 = identical distributions)")


def analyze_sequence_patterns(data, name):
    """Analyze sequence-level patterns."""
    events = data['event']
    
    print(f"\n{'='*60}")
    print(f"{name} - Sequence Patterns")
    print(f"{'='*60}")
    
    # Check for repetitive patterns
    num_traces = events.shape[0]
    seq_len = events.shape[1]
    
    # Count unique traces
    unique_traces = set()
    for trace in events:
        unique_traces.add(tuple(trace))
    
    print(f"Total traces: {num_traces}")
    print(f"Unique traces: {len(unique_traces)}")
    print(f"Diversity: {100*len(unique_traces)/num_traces:.2f}%")
    
    # Check for repeating subsequences within traces
    repeat_counts = []
    for trace in events[:100]:  # Sample first 100 traces
        # Check for immediate repeats (e.g., [1,1,1,1])
        repeats = 0
        for i in range(len(trace) - 1):
            if trace[i] == trace[i+1]:
                repeats += 1
        repeat_counts.append(repeats)
    
    avg_repeats = np.mean(repeat_counts)
    print(f"\nAverage immediate repeats per trace (first 100): {avg_repeats:.2f}")
    print(f"Max immediate repeats: {max(repeat_counts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-data', required=True)
    parser.add_argument('--synthetic-data', required=True)
    parser.add_argument('--output', default='experiments_downstream/data_quality_report.txt')
    
    args = parser.parse_args()
    
    print("Loading data...")
    real_data = np.load(args.real_data)
    synth_data = np.load(args.synthetic_data)
    
    # Event distribution
    real_events = analyze_event_distribution(real_data, "REAL DATA")
    synth_events = analyze_event_distribution(synth_data, "SYNTHETIC DATA")
    
    # Transitions
    real_trans = analyze_transitions(real_data, "REAL DATA")
    synth_trans = analyze_transitions(synth_data, "SYNTHETIC DATA")
    
    # Comparisons
    compare_distributions(real_events, synth_events, "events")
    compare_distributions(real_trans, synth_trans, "transitions")
    
    # Sequence patterns
    analyze_sequence_patterns(real_data, "REAL DATA")
    analyze_sequence_patterns(synth_data, "SYNTHETIC DATA")
    
    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
