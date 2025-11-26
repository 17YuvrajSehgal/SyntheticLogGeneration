"""
Diagnostic script to analyze generation issues.
"""

import numpy as np
from data_loader import TraceDataset

# Check real data statistics
print("=== Real Data Statistics ===")
real_dataset = TraceDataset('Datasets/compress-gzip/sequence_length_200/testing', 200, normalize=False)
real_data = real_dataset.data.flatten()
real_unique = np.unique(real_data)

print(f"Min event ID: {real_dataset.min_val}")
print(f"Max event ID: {real_dataset.max_val}")
print(f"Number of unique event IDs: {len(real_unique)}")
print(f"Unique event IDs: {sorted(real_unique)}")
print(f"Event ID range: {real_unique.min()} to {real_unique.max()}")

# Check what normalization values were used
print("\n=== Normalization Values (from training) ===")
import torch
checkpoint = torch.load('checkpoints/compress-gzip/best.pt', map_location='cpu')
if 'normalization' in checkpoint:
    norm = checkpoint['normalization']
    print(f"Min (from checkpoint): {norm.get('min_val')}")
    print(f"Max (from checkpoint): {norm.get('max_val')}")
else:
    print("No normalization values in checkpoint!")

# Check generated data
print("\n=== Generated Data Analysis ===")
try:
    with open('generated_traces/compress-gzip/generated_traces.txt', 'r') as f:
        gen_sequences = []
        for line in f:
            seq = [int(x) for x in line.strip().split(',')]
            gen_sequences.append(seq)
    
    gen_data = np.array(gen_sequences).flatten()
    gen_unique = np.unique(gen_data)
    
    print(f"Generated min value: {gen_data.min()}")
    print(f"Generated max value: {gen_data.max()}")
    print(f"Number of unique generated event IDs: {len(gen_unique)}")
    print(f"Generated event ID range: {gen_unique.min()} to {gen_unique.max()}")
    print(f"\nOverlap with real event IDs: {len(set(gen_unique) & set(real_unique))} / {len(real_unique)}")
    
    # Check distribution
    print(f"\nMost common generated event IDs:")
    from collections import Counter
    gen_counts = Counter(gen_data)
    for event, count in gen_counts.most_common(10):
        print(f"  Event {event}: {count} times ({count/len(gen_data)*100:.2f}%)")
        
except FileNotFoundError:
    print("Generated traces file not found")


