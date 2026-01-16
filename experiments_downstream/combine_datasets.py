"""
Combine real and synthetic datasets for downstream experiments.

Usage:
    python experiments_downstream/combine_datasets.py \
        --real-data path/to/real_train.npz \
        --synthetic-data path/to/synthetic_repaired.npz \
        --output path/to/combined.npz \
        --ratio 0.5
"""

import argparse
import numpy as np
import os


def combine_datasets(real_path, synthetic_path, output_path, real_ratio=0.5, seed=42):
    """
    Combine real and synthetic datasets.
    
    Args:
        real_path: Path to real training data (.npz)
        synthetic_path: Path to synthetic data (.npz)
        output_path: Path to save combined data (.npz)
        real_ratio: Ratio of real data (0.5 = 50% real, 50% synthetic)
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print("Combining Datasets")
    print(f"{'='*60}")
    
    # Load data
    print(f"[Load] Real data: {real_path}")
    real_data = np.load(real_path)
    
    print(f"[Load] Synthetic data: {synthetic_path}")
    synth_data = np.load(synthetic_path)
    
    # Get sizes
    num_real = real_data['event'].shape[0]
    num_synth = synth_data['event'].shape[0]
    real_seq_len = real_data['event'].shape[1]
    synth_seq_len = synth_data['event'].shape[1]
    
    print(f"\n[Info] Real data: {num_real} traces, seq_len={real_seq_len}")
    print(f"[Info] Synthetic data: {num_synth} traces, seq_len={synth_seq_len}")
    
    # Check sequence length compatibility
    if real_seq_len != synth_seq_len:
        raise ValueError(
            f"Sequence length mismatch: real={real_seq_len}, synthetic={synth_seq_len}. "
            f"Cannot combine datasets with different sequence lengths."
        )
    
    # Calculate how many samples to take from each
    total_samples = min(num_real, num_synth) * 2  # Balanced
    num_real_samples = int(total_samples * real_ratio)
    num_synth_samples = total_samples - num_real_samples
    
    print(f"\n[Combine] Mixing with ratio {real_ratio:.0%} real / {1-real_ratio:.0%} synthetic")
    print(f"[Combine] Taking {num_real_samples} real + {num_synth_samples} synthetic = {total_samples} total")
    
    # Sample
    np.random.seed(seed)
    real_indices = np.random.choice(num_real, num_real_samples, replace=False)
    synth_indices = np.random.choice(num_synth, num_synth_samples, replace=False)
    
    # Combine
    combined_data = {}
    for key in real_data.files:
        if key in synth_data.files:
            combined_data[key] = np.concatenate([
                real_data[key][real_indices],
                synth_data[key][synth_indices]
            ], axis=0)
            print(f"  - Combined '{key}': {combined_data[key].shape}")
        else:
            print(f"  - Skipped '{key}': not in synthetic data")
    
    # Shuffle
    print(f"\n[Shuffle] Shuffling combined data...")
    num_total = combined_data['event'].shape[0]
    shuffle_indices = np.random.permutation(num_total)
    combined_data = {k: v[shuffle_indices] for k, v in combined_data.items()}
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"\n[Save] Saving to: {output_path}")
    np.savez_compressed(output_path, **combined_data)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"Combined dataset: {output_path}")
    print(f"Total samples: {num_total}")
    print(f"Ratio: {num_real_samples}/{num_total} real ({100*num_real_samples/num_total:.1f}%), "
          f"{num_synth_samples}/{num_total} synthetic ({100*num_synth_samples/num_total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Combine real and synthetic datasets")
    
    parser.add_argument('--real-data', required=True, help='Path to real training data (.npz)')
    parser.add_argument('--synthetic-data', required=True, help='Path to synthetic data (.npz)')
    parser.add_argument('--output', required=True, help='Path to save combined data (.npz)')
    parser.add_argument('--ratio', type=float, default=0.5, 
                       help='Ratio of real data (default: 0.5 for 50/50 split)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratio
    if not 0 < args.ratio < 1:
        raise ValueError(f"Ratio must be between 0 and 1, got {args.ratio}")
    
    combine_datasets(
        args.real_data,
        args.synthetic_data,
        args.output,
        args.ratio,
        args.seed
    )


if __name__ == '__main__':
    main()
