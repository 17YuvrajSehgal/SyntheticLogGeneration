"""
Generation script using official SSSD model for unconditional trace generation.
Adapts the official SSSD inference code for generating new sequences from scratch.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

# Add project root to path for our modules
project_root = os.path.join(os.path.dirname(__file__), '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our data loader BEFORE changing directories
from data_loader import TraceDataset

# Add SSSD src to path BEFORE importing
sssd_src_path = os.path.join(os.path.dirname(__file__), '..', 'SSSD', 'src')
sssd_src_path = os.path.abspath(sssd_src_path)

if not os.path.exists(sssd_src_path):
    raise FileNotFoundError(f"SSSD src directory not found at: {sssd_src_path}\n"
                          f"Please ensure the SSSD repository is cloned in the SSSD/ directory.")

# Add to sys.path
if sssd_src_path not in sys.path:
    sys.path.insert(0, sssd_src_path)

# Import SSSD modules - they use relative imports from src directory
# We need to temporarily change working directory and ensure path is set
original_cwd = os.getcwd()

# Ensure sssd_src_path is first in sys.path
if sssd_src_path in sys.path:
    sys.path.remove(sssd_src_path)
sys.path.insert(0, sssd_src_path)

# Change to sssd_src_path for imports (SSSD code expects to be run from src/)
os.chdir(sssd_src_path)

try:
    # Clear any cached imports that might interfere
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('imputers.') or k.startswith('utils.')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # Now import - these should work with relative imports
    from imputers.SSSDS4Imputer import SSSDS4Imputer
    from utils.util import find_max_epoch, calc_diffusion_hyperparams, print_size, sampling
finally:
    # Restore original directory
    os.chdir(original_cwd)


def generate_unconditional(output_directory,
                          ckpt_path,
                          ckpt_iter,
                          num_samples,
                          sequence_length=200,
                          batch_size=16):
    """
    Generate unconditional sequences using trained SSSDS4 model.
    
    Parameters:
    output_directory (str):     where to save generated sequences
    ckpt_path (str):            path to checkpoint directory
    ckpt_iter (int or 'max'):   which checkpoint to load
    num_samples (int):          number of sequences to generate
    sequence_length (int):      length of each sequence
    batch_size (int):           batch size for generation
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint to get config
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    
    # Find checkpoint file
    checkpoint_files = [f for f in os.listdir(ckpt_path) if f.endswith('.pkl')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_path}")
    
    # Load the checkpoint
    checkpoint_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    if not os.path.exists(checkpoint_path):
        # Try to find any checkpoint
        checkpoint_path = os.path.join(ckpt_path, checkpoint_files[0])
        print(f"Warning: Specified checkpoint not found, using {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Get configs from checkpoint
    model_config = checkpoint.get('model_config', {})
    diffusion_config = checkpoint.get('diffusion_config', {'T': 200, 'beta_0': 0.0001, 'beta_T': 0.02})
    norm_info = checkpoint.get('normalization', {})
    
    # Calculate diffusion hyperparameters
    T = diffusion_config['T']
    beta_0 = diffusion_config['beta_0']
    beta_T = diffusion_config['beta_T']
    diffusion_hyperparams = calc_diffusion_hyperparams(T, beta_0, beta_T)
    
    # Map to device
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)
    
    # Create model
    print("Creating model...")
    net = SSSDS4Imputer(**model_config).to(device)
    print_size(net)
    
    # Load weights
    # Handle shared memory issues with S4 layer parameters
    # The S4 kernel parameters (B, P, w) have shared memory in the model
    # We need to load with strict=False and manually set these parameters
    state_dict = checkpoint['model_state_dict']
    
    # First, load all non-problematic parameters
    try:
        net.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        # If that fails, manually load parameters one by one
        model_dict = net.state_dict()
        for key, value in state_dict.items():
            if key in model_dict:
                # Get the parameter object from the model
                param = dict(net.named_parameters())[key] if key in dict(net.named_parameters()) else None
                if param is not None:
                    # For shared memory parameters, replace the entire parameter
                    try:
                        with torch.no_grad():
                            param.data = value.clone().detach().to(param.device)
                    except RuntimeError as e:
                        if "more than one element" in str(e):
                            # Create a new parameter without shared memory
                            new_param = nn.Parameter(value.clone().detach().to(param.device))
                            # Replace the parameter in the module
                            parts = key.split('.')
                            module = net
                            for part in parts[:-1]:
                                module = getattr(module, part)
                            setattr(module, parts[-1], new_param)
                        else:
                            raise
    
    net.eval()
    print('Model loaded successfully')
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate sequences
    print(f"Generating {num_samples} sequences of length {sequence_length}...")
    all_generated = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Start with pure noise
            # Shape: (batch_size, channels, seq_len)
            noise = torch.randn(current_batch_size, 1, sequence_length).to(device)
            
            # Create a mask that marks everything as missing (for unconditional generation)
            # This tells the model to generate the entire sequence
            mask = torch.ones(current_batch_size, 1, sequence_length).to(device)
            
            # Generate using diffusion sampling
            # For unconditional generation, cond is the same as noise (we'll replace with generated values)
            # The sampling function signature: sampling(net, size, diffusion_hyperparams, cond, mask, ...)
            cond = noise.clone()  # Initial condition (will be updated during sampling)
            generated = sampling(net, noise.shape, diffusion_hyperparams, cond, mask, only_generate_missing=1)
            
            all_generated.append(generated.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Generated {min((i + 1) * batch_size, num_samples)} / {num_samples} sequences")
    
    # Concatenate all batches
    generated_sequences = np.concatenate(all_generated, axis=0)[:num_samples]
    # Shape: (num_samples, 1, seq_len) -> (num_samples, seq_len)
    generated_sequences = generated_sequences.squeeze(1)
    
    # Denormalize (Bug 2 fix: don't use arbitrary defaults)
    train_min = norm_info.get('train_min')
    train_max = norm_info.get('train_max')
    
    if train_min is None or train_max is None:
        print("ERROR: Normalization information missing from checkpoint!")
        print("Cannot denormalize generated sequences. Please ensure checkpoint contains normalization info.")
        print("Skipping denormalization - generated values will be in [0, 1] range.")
        print("You may need to manually denormalize or retrain with normalization info saved.")
    elif train_max > train_min:
        generated_sequences = generated_sequences * (train_max - train_min) + train_min
        print(f"Denormalized using range [{train_min}, {train_max}]")
    else:
        print("Warning: Invalid normalization range (max <= min), skipping denormalization")
    
    # Quantize to integers
    generated_sequences = np.round(generated_sequences).astype(np.int32)
    
    # Constrain to valid event IDs
    # Load real data to get valid event IDs
    try:
        dataset_name = os.path.basename(os.path.dirname(ckpt_path))
        test_data_path = f"Datasets/{dataset_name}/sequence_length_{sequence_length}/testing"
        real_dataset = TraceDataset(test_data_path, sequence_length, normalize=False)
        real_unique = np.unique(real_dataset.data.flatten())
        valid_min = int(real_dataset.min_val)
        valid_max = int(real_dataset.max_val)
        
        print(f"Valid event ID range: {valid_min} to {valid_max}")
        print(f"Number of valid event IDs: {len(real_unique)}")
        
        # Clip to valid range
        generated_sequences = np.clip(generated_sequences, valid_min, valid_max)
        
        # Map to nearest valid event ID
        print("Mapping to nearest valid event IDs...")
        for i in range(len(generated_sequences)):
            for j in range(len(generated_sequences[i])):
                val = generated_sequences[i, j]
                nearest_idx = np.argmin(np.abs(real_unique - val))
                generated_sequences[i, j] = int(real_unique[nearest_idx])
        
        print(f"After mapping: {len(np.unique(generated_sequences))} unique event IDs")
        
    except Exception as e:
        print(f"Warning: Could not load real data for validation: {e}")
        generated_sequences = np.clip(generated_sequences, 0, None)
    
    # Save generated sequences
    output_file = os.path.join(output_directory, 'generated_traces.txt')
    print(f"Saving to {output_file}")
    with open(output_file, 'w') as f:
        for seq in generated_sequences:
            seq_str = ','.join(map(str, seq))
            f.write(seq_str + '\n')
    
    # Save as numpy array
    np.save(os.path.join(output_directory, 'generated_traces.npy'), generated_sequences)
    
    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'seq_len': sequence_length,
        'checkpoint': checkpoint_path,
        'checkpoint_iter': ckpt_iter,
        'normalization': norm_info
    }
    with open(os.path.join(output_directory, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {num_samples} sequences")
    print(f"Output saved to {output_directory}")
    print(f"Sequence statistics:")
    print(f"  Min value: {generated_sequences.min()}")
    print(f"  Max value: {generated_sequences.max()}")
    print(f"  Mean value: {generated_sequences.mean():.2f}")
    print(f"  Unique event IDs: {len(np.unique(generated_sequences))}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic traces using SSSDS4')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--checkpoint_iter', type=str, default='max', help='Checkpoint iteration to load')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of sequences to generate')
    parser.add_argument('--seq_len', type=int, default=200, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='generated_traces_sssd', help='Output directory')
    
    args = parser.parse_args()
    
    generate_unconditional(
        output_directory=args.output_dir,
        ckpt_path=args.checkpoint,
        ckpt_iter=args.checkpoint_iter,
        num_samples=args.num_samples,
        sequence_length=args.seq_len,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

