"""
Generation script for creating synthetic kernel traces using trained SSSDS4 model.
"""

import torch
import numpy as np
import argparse
import os
import json
from tqdm import tqdm

from models.sssds4 import SSSDS4, DiffusionScheduler, TraceGenerator


def denormalize(data, min_val, max_val):
    """Denormalize data back to original scale."""
    if max_val > min_val:
        return data * (max_val - min_val) + min_val
    return data


def quantize_to_integers(data):
    """Quantize continuous values to nearest integers (system call IDs)."""
    return np.round(data).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic kernel traces')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='generated_traces',
                       help='Directory to save generated traces')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of sequences to generate (default: 100)')
    parser.add_argument('--seq_len', type=int, default=200,
                       help='Length of generated sequences (default: 200)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for generation (default: 16)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--output_format', type=str, default='text',
                       choices=['text', 'npy', 'both'],
                       help='Output format (default: text)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model arguments
    model_args = checkpoint.get('args', {})
    d_model = model_args.get('d_model', 128)
    num_layers = model_args.get('num_layers', 4)
    num_timesteps = model_args.get('num_timesteps', 1000)
    
    # Create model
    model = SSSDS4(
        input_dim=1,
        d_model=d_model,
        num_layers=num_layers,
        num_timesteps=num_timesteps
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Create scheduler and generator
    scheduler = DiffusionScheduler(num_timesteps=num_timesteps)
    generator = TraceGenerator(model, scheduler, device)
    
    # Generate traces
    print(f"Generating {args.num_samples} sequences of length {args.seq_len}...")
    
    all_generated = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(num_batches)):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        generated = generator.generate(
            seq_len=args.seq_len,
            batch_size=batch_size,
            num_samples=1
        )
        all_generated.append(generated)
    
    # Concatenate all generated sequences
    generated_sequences = np.concatenate(all_generated, axis=0)[:args.num_samples]
    
    # Remove channel dimension and quantize to integers
    generated_sequences = generated_sequences.squeeze(-1)  # (num_samples, seq_len)
    
    # Denormalize (assuming normalization was used during training)
    # Try to load min/max values from checkpoint metadata
    min_val = None
    max_val = None
    
    # Check if normalization info is in checkpoint
    if 'normalization' in checkpoint:
        min_val = checkpoint['normalization'].get('min_val')
        max_val = checkpoint['normalization'].get('max_val')
    
    # If not in checkpoint, try to infer from dataset
    if min_val is None or max_val is None:
        # Try to load from dataset to get actual range
        dataset_name = model_args.get('dataset', '')
        if dataset_name:
            try:
                from data_loader import TraceDataset
                dataset_path = f"Datasets/{dataset_name}/sequence_length_{args.seq_len}/training"
                if os.path.exists(dataset_path):
                    temp_dataset = TraceDataset(dataset_path, args.seq_len, normalize=False)
                    min_val = temp_dataset.min_val
                    max_val = temp_dataset.max_val
                    print(f"Loaded normalization values from dataset: min={min_val}, max={max_val}")
            except:
                pass
    
    # Fallback: use reasonable defaults (adjust based on your data)
    if min_val is None:
        min_val = 0
    if max_val is None:
        max_val = 100  # Adjust based on your data (max system call ID)
        print(f"Warning: Using default normalization values. min={min_val}, max={max_val}")
        print("Consider saving normalization values during training for better results.")
    
    generated_sequences = denormalize(generated_sequences, min_val, max_val)
    generated_sequences = quantize_to_integers(generated_sequences)
    
    # Get actual valid event ID range from real data
    try:
        from data_loader import TraceDataset
        dataset_name = model_args.get('dataset', 'compress-gzip')
        real_dataset = TraceDataset(
            f"Datasets/{dataset_name}/sequence_length_{args.seq_len}/testing",
            args.seq_len,
            normalize=False
        )
        valid_min = int(real_dataset.min_val)
        valid_max = int(real_dataset.max_val)
        real_unique = np.unique(real_dataset.data.flatten())
        
        print(f"Valid event ID range: {valid_min} to {valid_max}")
        print(f"Number of valid event IDs: {len(real_unique)}")
        print(f"Valid event IDs: {sorted(real_unique)}")
        
        # Clip to valid range first
        generated_sequences = np.clip(generated_sequences, valid_min, valid_max)
        
        # Map to nearest valid event ID (ensures we only generate IDs that exist in real data)
        # This is important for discrete event sequences
        print("Mapping generated values to nearest valid event IDs...")
        for i in range(len(generated_sequences)):
            for j in range(len(generated_sequences[i])):
                val = generated_sequences[i, j]
                # Find nearest valid event ID
                nearest_idx = np.argmin(np.abs(real_unique - val))
                generated_sequences[i, j] = int(real_unique[nearest_idx])
        
        print(f"After mapping: {len(np.unique(generated_sequences))} unique event IDs in generated data")
        
    except Exception as e:
        print(f"Warning: Could not load real data for validation: {e}")
        print("Clipping to positive values only...")
        generated_sequences = np.clip(generated_sequences, 0, None)
    
    # Save generated traces
    if args.output_format in ['text', 'both']:
        output_file = os.path.join(args.output_dir, 'generated_traces.txt')
        print(f"Saving to {output_file}")
        with open(output_file, 'w') as f:
            for seq in generated_sequences:
                seq_str = ','.join(map(str, seq))
                f.write(seq_str + '\n')
    
    if args.output_format in ['npy', 'both']:
        output_file = os.path.join(args.output_dir, 'generated_traces.npy')
        print(f"Saving to {output_file}")
        np.save(output_file, generated_sequences)
    
    # Save metadata
    metadata = {
        'num_samples': args.num_samples,
        'seq_len': args.seq_len,
        'checkpoint': args.checkpoint,
        'model_epoch': checkpoint['epoch'],
        'model_args': model_args
    }
    
    metadata_file = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {args.num_samples} sequences")
    print(f"Output saved to {args.output_dir}")
    print(f"\nSequence statistics:")
    print(f"  Min value: {generated_sequences.min()}")
    print(f"  Max value: {generated_sequences.max()}")
    print(f"  Mean value: {generated_sequences.mean():.2f}")
    print(f"  Std value: {generated_sequences.std():.2f}")
    print(f"  Unique event IDs: {len(np.unique(generated_sequences))}")
    
    # Show distribution of generated event IDs
    from collections import Counter
    gen_counts = Counter(generated_sequences.flatten())
    print(f"\nTop 10 most common generated event IDs:")
    for event, count in gen_counts.most_common(10):
        print(f"  Event {event}: {count} times ({count/len(generated_sequences.flatten())*100:.2f}%)")


if __name__ == "__main__":
    main()

