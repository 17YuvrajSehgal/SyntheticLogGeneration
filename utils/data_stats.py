"""
Utility script to analyze dataset statistics.
Helps understand the data distribution before training.
"""

import numpy as np
import argparse
import os
from collections import Counter


def analyze_dataset(data_path):
    """Analyze statistics of a trace dataset."""
    sequences = []
    all_events = []
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        for line in f:
            seq = [int(x) for x in line.strip().split(',')]
            sequences.append(seq)
            all_events.extend(seq)
    
    # Basic statistics
    num_sequences = len(sequences)
    seq_lengths = [len(seq) for seq in sequences]
    event_counts = Counter(all_events)
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Number of sequences: {num_sequences}")
    print(f"Sequence length - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {np.mean(seq_lengths):.2f}")
    print(f"Total events: {len(all_events)}")
    print(f"Unique events: {len(event_counts)}")
    print(f"Event value range: {min(all_events)} - {max(all_events)}")
    
    # Event frequency
    print(f"\n=== Top 10 Most Frequent Events ===")
    for event, count in event_counts.most_common(10):
        percentage = (count / len(all_events)) * 100
        print(f"Event {event}: {count} occurrences ({percentage:.2f}%)")
    
    # Sequence patterns
    print(f"\n=== Sequence Pattern Analysis ===")
    print(f"Average sequence length: {np.mean(seq_lengths):.2f}")
    print(f"Std sequence length: {np.std(seq_lengths):.2f}")
    
    # Sample sequences
    print(f"\n=== Sample Sequences (first 3) ===")
    for i, seq in enumerate(sequences[:3]):
        print(f"Sequence {i+1} (length {len(seq)}): {seq[:20]}..." if len(seq) > 20 else f"Sequence {i+1}: {seq}")
    
    return {
        'num_sequences': num_sequences,
        'seq_lengths': seq_lengths,
        'event_counts': event_counts,
        'min_event': min(all_events),
        'max_event': max(all_events),
        'num_unique_events': len(event_counts)
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset statistics')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file (training or testing)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: File {args.data_path} not found")
        return
    
    stats = analyze_dataset(args.data_path)
    
    # Save statistics
    output_file = args.data_path + '_stats.txt'
    with open(output_file, 'w') as f:
        f.write(f"Dataset Statistics\n")
        f.write(f"==================\n\n")
        f.write(f"Number of sequences: {stats['num_sequences']}\n")
        f.write(f"Sequence length - Min: {min(stats['seq_lengths'])}, Max: {max(stats['seq_lengths'])}, Mean: {np.mean(stats['seq_lengths']):.2f}\n")
        f.write(f"Unique events: {stats['num_unique_events']}\n")
        f.write(f"Event value range: {stats['min_event']} - {stats['max_event']}\n")
    
    print(f"\nStatistics saved to {output_file}")


if __name__ == "__main__":
    main()

