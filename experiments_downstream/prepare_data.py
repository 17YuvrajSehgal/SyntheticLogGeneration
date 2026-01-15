"""
Prepare data for downstream task experiments.

This script:
1. Splits real data into train/test
2. Generates synthetic data (if needed)
3. Repairs synthetic data
4. Creates combined datasets
"""

import argparse
import os
import numpy as np
import subprocess
from pathlib import Path


def split_real_data(input_glob, output_dir, train_ratio=0.8, seed=42):
    """Split real data into train/test sets."""
    import glob
    
    print(f"[Split] Finding files matching: {input_glob}")
    files = sorted(glob.glob(input_glob, recursive=True))
    
    if not files:
        raise FileNotFoundError(f"No files found matching: {input_glob}")
    
    print(f"[Split] Found {len(files)} files")
    
    # Load all data
    all_data = {}
    for f in files:
        data = np.load(f)
        for key in data.files:
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(data[key])
    
    # Concatenate
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key], axis=0)
        print(f"[Split] {key}: {all_data[key].shape}")
    
    # Shuffle
    np.random.seed(seed)
    num_samples = all_data['event'].shape[0]
    indices = np.random.permutation(num_samples)
    
    split_idx = int(num_samples * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    print(f"[Split] Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    # Save train
    train_data = {k: v[train_indices] for k, v in all_data.items()}
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, 'real_train.npz'), **train_data)
    print(f"[Saved] {os.path.join(output_dir, 'real_train.npz')}")
    
    # Save test
    test_data = {k: v[test_indices] for k, v in all_data.items()}
    np.savez_compressed(os.path.join(output_dir, 'real_test.npz'), **test_data)
    print(f"[Saved] {os.path.join(output_dir, 'real_test.npz')}")
    
    return os.path.join(output_dir, 'real_train.npz'), os.path.join(output_dir, 'real_test.npz')


def generate_synthetic(checkpoint, output_path, num_samples=10000, seq_len=1024):
    """Generate synthetic traces using sample_diffusion.py."""
    print(f"\n[Generate] Generating {num_samples} synthetic samples...")
    print(f"[Generate] Checkpoint: {checkpoint}")
    
    # Optimized batch sizes for maximum speed on H100 GPU (80GB)
    # Larger batches for shorter sequences = faster generation
    if seq_len <= 256:
        batch_size = 64  # Very fast for short sequences
    elif seq_len <= 1024:
        batch_size = 32  # Optimal for medium sequences
    elif seq_len <= 2048:
        batch_size = 16  # Good balance
    else:  # 4096
        batch_size = 8   # Necessary for long sequences
    
    print(f"[Generate] Using optimized batch size: {batch_size} for seq_len={seq_len}")
    
    cmd = [
        'python', 'sample_diffusion.py',
        '--ckpt', checkpoint,
        '--out', output_path,
        '--num-samples', str(num_samples),
        '--seq-len', str(seq_len),
        '--batch-size', str(batch_size),
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    print(f"[Saved] {output_path}")
    return output_path


def repair_synthetic(input_path, constraints_path, output_path):
    """Repair synthetic traces using repair.py."""
    print(f"\n[Repair] Repairing {input_path}...")
    
    cmd = [
        'python', 'synthetic_log_gen/repair.py',
        '--trace', input_path,
        '--constraints', constraints_path,
        '--output', output_path,
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    print(f"[Saved] {output_path}")
    return output_path


def combine_datasets(real_path, synthetic_path, output_path, real_ratio=0.5, seed=42):
    """Combine real and synthetic data."""
    print(f"\n[Combine] Mixing real and synthetic data (ratio={real_ratio})...")
    
    # Load data
    real_data = np.load(real_path)
    synth_data = np.load(synthetic_path)
    
    # Determine sizes
    num_real = real_data['event'].shape[0]
    num_synth = synth_data['event'].shape[0]
    
    # Calculate how many samples to take from each
    total_samples = min(num_real, num_synth) * 2  # Balanced
    num_real_samples = int(total_samples * real_ratio)
    num_synth_samples = total_samples - num_real_samples
    
    print(f"[Combine] Real: {num_real_samples}, Synthetic: {num_synth_samples}")
    
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
    
    # Shuffle
    num_total = combined_data['event'].shape[0]
    shuffle_indices = np.random.permutation(num_total)
    combined_data = {k: v[shuffle_indices] for k, v in combined_data.items()}
    
    # Save
    np.savez_compressed(output_path, **combined_data)
    print(f"[Saved] {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    
    # Input
    parser.add_argument('--real-glob', required=True, help='Glob pattern for real data')
    parser.add_argument('--benchmark', default='scimark2')
    
    # Synthetic generation (optional)
    parser.add_argument('--generate-synthetic', action='store_true')
    parser.add_argument('--checkpoint-256', help='Checkpoint for context_256 model')
    parser.add_argument('--checkpoint-1024', help='Checkpoint for context_1024 model')
    parser.add_argument('--checkpoint-4096', help='Checkpoint for context_4096 model')
    parser.add_argument('--num-synthetic-samples', type=int, default=10000)
    
    # Repair
    parser.add_argument('--constraints', default='dataset/constraints_universal.json')
    
    # Output
    parser.add_argument('--output-dir', default='experiments_downstream/data')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Split real data
    print("="*60)
    print("STEP 1: Splitting Real Data")
    print("="*60)
    
    real_train, real_test = split_real_data(
        args.real_glob,
        output_dir,
        train_ratio=0.8
    )
    
    # Step 2: Generate synthetic data (if requested)
    if args.generate_synthetic:
        print("\n" + "="*60)
        print("STEP 2: Generating Synthetic Data")
        print("="*60)
        
        checkpoints = {
            '256': args.checkpoint_256,
            '1024': args.checkpoint_1024,
            '4096': args.checkpoint_4096,
        }
        
        synthetic_files = {}
        
        for context, ckpt in checkpoints.items():
            if ckpt and os.path.exists(ckpt):
                output_path = os.path.join(output_dir, f'synthetic_raw_{context}.npz')
                synthetic_files[context] = generate_synthetic(
                    ckpt,
                    output_path,
                    num_samples=args.num_synthetic_samples,
                    seq_len=int(context)
                )
    else:
        print("\n[Skip] Synthetic generation (use --generate-synthetic to enable)")
        # Assume files already exist
        synthetic_files = {
            '256': os.path.join(output_dir, 'synthetic_raw_256.npz'),
            '1024': os.path.join(output_dir, 'synthetic_raw_1024.npz'),
            '4096': os.path.join(output_dir, 'synthetic_raw_4096.npz'),
        }
    
    # Step 3: Repair synthetic data
    print("\n" + "="*60)
    print("STEP 3: Repairing Synthetic Data")
    print("="*60)
    
    repaired_files = {}
    for context, synth_file in synthetic_files.items():
        if os.path.exists(synth_file):
            output_path = os.path.join(output_dir, f'synthetic_repaired_{context}.npz')
            repaired_files[context] = repair_synthetic(
                synth_file,
                args.constraints,
                output_path
            )
    
    # Step 4: Create combined datasets
    print("\n" + "="*60)
    print("STEP 4: Creating Combined Datasets")
    print("="*60)
    
    for context, repaired_file in repaired_files.items():
        if os.path.exists(repaired_file):
            output_path = os.path.join(output_dir, f'combined_real_synthetic_{context}.npz')
            combine_datasets(
                real_train,
                repaired_file,
                output_path,
                real_ratio=0.5
            )
    
    print("\n" + "="*60)
    print("DONE: Data Preparation Complete")
    print("="*60)
    print(f"\nData saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Train baseline: python experiments_downstream/models/train_predictor.py --train-data data/real_train.npz --test-data data/real_test.npz --run-name real_baseline")
    print("2. Train on synthetic: python experiments_downstream/models/train_predictor.py --train-data data/synthetic_repaired_1024.npz --test-data data/real_test.npz --run-name synthetic_1024")
    print("3. Train on combined: python experiments_downstream/models/train_predictor.py --train-data data/combined_real_synthetic_1024.npz --test-data data/real_test.npz --run-name combined_1024")


if __name__ == '__main__':
    main()
