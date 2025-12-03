"""
Training script using official SSSD repository for kernel trace generation.
Adapts the official SSSD training code to work with our trace data format.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from SSSD.src.imputers.SSSDS4Imputer import SSSDS4Imputer
from SSSD.src.utils.util import calc_diffusion_hyperparams, print_size, find_max_epoch, get_mask_rm, get_mask_bm, \
    get_mask_mnr, training_loss

# Add SSSD src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SSSD', 'src'))

# from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
# from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
# from imputers.SSSDS4Imputer import SSSDS4Imputer
from data_loader import TraceDataset
from torch.utils.data import DataLoader


def load_trace_data(data_path, sequence_length=200, normalize=True):
    """Load trace data and convert to format expected by SSSD."""
    dataset = TraceDataset(data_path, sequence_length, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Convert to numpy array: (num_samples, seq_len, channels)
    # For trace data, we treat each sequence as a single channel
    sequences = []
    for batch in dataloader:
        seq = batch.numpy()  # (1, seq_len)
        sequences.append(seq[0])  # (seq_len,)
    
    # Reshape to (num_samples, channels, seq_len) for SSSD
    # We treat each sequence as having 1 channel
    data = np.array(sequences)  # (num_samples, seq_len)
    data = data[:, np.newaxis, :]  # (num_samples, 1, seq_len)
    
    return data, dataset.min_val, dataset.max_val


def train(output_directory,
          train_data_path,
          test_data_path,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          sequence_length=200,
          batch_size=32,
          masking='rm',
          missing_k=10):
    """
    Train SSSDS4 model on kernel trace data.
    
    Parameters:
    output_directory (str):         save model checkpoints to this path
    train_data_path (str):          path to training data
    test_data_path (str):           path to test data
    ckpt_iter (int or 'max'):       checkpoint to resume from
    n_iters (int):                  number of training iterations
    iters_per_ckpt (int):           save checkpoint every N iterations
    iters_per_logging (int):        log every N iterations
    learning_rate (float):          learning rate
    sequence_length (int):          length of sequences
    batch_size (int):               batch size
    masking (str):                  'mnr', 'bm', or 'rm'
    missing_k (int):                number of missing points
    """
    
    # Diffusion hyperparameters
    T = 200
    beta_0 = 0.0001
    beta_T = 0.02
    
    diffusion_config = {
        "T": T,
        "beta_0": beta_0,
        "beta_T": beta_T
    }
    
    diffusion_hyperparams = calc_diffusion_hyperparams(T, beta_0, beta_T)
    
    # Model configuration for trace data (1 channel, sequence length)
    model_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_res_layers": 24,  # Reduced from 36 for faster training
        "res_channels": 128,   # Reduced from 256
        "skip_channels": 128,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": sequence_length,
        "s4_d_state": 64,
        "s4_dropout": 0.0,
        "s4_bidirectional": 1,
        "s4_layernorm": 1
    }
    
    # Generate experiment path
    local_path = "T{}_beta0{}_betaT{}".format(T, beta_0, beta_T)
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    print("Output directory:", output_directory, flush=True)
    
    # Map diffusion hyperparameters to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)
    
    # Load data
    print("Loading training data...")
    train_data, train_min, train_max = load_trace_data(train_data_path, sequence_length, normalize=True)
    print(f"Training data shape: {train_data.shape}")
    print(f"Data range: [{train_min}, {train_max}]")
    
    print("Loading test data...")
    test_data, test_min, test_max = load_trace_data(test_data_path, sequence_length, normalize=True)
    print(f"Test data shape: {test_data.shape}")
    
    # Save normalization info
    norm_info = {
        'train_min': float(train_min),
        'train_max': float(train_max),
        'test_min': float(test_min),
        'test_max': float(test_max)
    }
    with open(os.path.join(output_directory, 'normalization.json'), 'w') as f:
        json.dump(norm_info, f, indent=2)
    
    # Convert to torch tensors
    train_data = torch.from_numpy(train_data).float().to(device)
    test_data = torch.from_numpy(test_data).float().to(device)
    
    # Create model
    print("Creating model...")
    net = SSSDS4Imputer(**model_config).to(device)
    print_size(net)
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    # Load checkpoint if resuming
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint found, starting from scratch.')
    else:
        ckpt_iter = -1
    
    # Training loop
    print("Starting training...")
    net.train()
    
    for n in range(ckpt_iter + 1, n_iters):
        # Sample random batch
        batch_indices = np.random.choice(len(train_data), size=batch_size, replace=False)
        batch = train_data[batch_indices]  # (batch_size, 1, seq_len)
        
        # Get mask based on masking strategy
        # The mask functions work on individual samples, so we need to create masks for each sample
        masks = []
        for i in range(batch_size):
            sample = batch[i, 0, :].cpu()  # (seq_len,) - single channel
            if masking == 'rm':
                sample_mask = get_mask_rm(sample, missing_k)  # Returns (seq_len, 1)
            elif masking == 'bm':
                sample_mask = get_mask_bm(sample, missing_k)
            elif masking == 'mnr':
                sample_mask = get_mask_mnr(sample, missing_k)
            else:
                raise ValueError(f"Unknown masking strategy: {masking}")
            masks.append(sample_mask)
        
        # Stack masks: (batch_size, seq_len, 1) -> (batch_size, 1, seq_len)
        mask = torch.stack(masks, dim=0).to(device)  # (batch_size, seq_len, 1)
        mask = mask.permute(0, 2, 1)  # (batch_size, 1, seq_len)
        
        # Training step - prepare data in format expected by training_loss
        loss_mask = ~mask.bool()  # Invert mask for loss calculation
        X = (batch, batch, mask, loss_mask)  # (audio, cond, mask, loss_mask)
        loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams, only_generate_missing=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if n % iters_per_logging == 0:
            print("iteration: {} \tloss: {}".format(n, loss.item()), flush=True)
            
            # Validation
            net.eval()
            with torch.no_grad():
                val_indices = np.random.choice(len(test_data), size=min(batch_size, len(test_data)), replace=False)
                val_batch = test_data[val_indices]
                
                # Create masks for validation batch
                val_masks = []
                for i in range(len(val_batch)):
                    sample = val_batch[i, 0, :].cpu()
                    val_sample_mask = get_mask_rm(sample, missing_k)
                    val_masks.append(val_sample_mask)
                val_mask = torch.stack(val_masks, dim=0).to(device)
                val_mask = val_mask.permute(0, 2, 1)  # (batch_size, 1, seq_len)
                
                val_loss_mask = ~val_mask.bool()
                val_X = (val_batch, val_batch, val_mask, val_loss_mask)
                val_loss = training_loss(net, nn.MSELoss(), val_X, diffusion_hyperparams, only_generate_missing=1)
                print("validation loss: {}".format(val_loss.item()), flush=True)
            net.train()
        
        # Save checkpoint
        if n % iters_per_ckpt == 0:
            checkpoint = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': n,
                'model_config': model_config,
                'diffusion_config': diffusion_config,
                'normalization': norm_info
            }
            checkpoint_path = os.path.join(output_directory, '{}.pkl'.format(n))
            torch.save(checkpoint, checkpoint_path)
            print('model at iteration %s is saved' % n, flush=True)
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train SSSDS4 on kernel trace data')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--sequence_length', type=int, default=200, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_iters', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--iters_per_ckpt', type=int, default=1000, help='Save checkpoint every N iterations')
    parser.add_argument('--iters_per_logging', type=int, default=100, help='Log every N iterations')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--masking', type=str, default='rm', choices=['rm', 'bm', 'mnr'], help='Masking strategy')
    parser.add_argument('--missing_k', type=int, default=10, help='Number of missing points')
    parser.add_argument('--output_dir', type=str, default='checkpoints_sssd', help='Output directory')
    parser.add_argument('--resume', type=str, default='max', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train_data_path = f"Datasets/{args.dataset}/sequence_length_{args.sequence_length}/training"
    test_data_path = f"Datasets/{args.dataset}/sequence_length_{args.sequence_length}/testing"
    output_dir = os.path.join(args.output_dir, args.dataset)
    
    train(
        output_directory=output_dir,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        ckpt_iter=args.resume,
        n_iters=args.n_iters,
        iters_per_ckpt=args.iters_per_ckpt,
        iters_per_logging=args.iters_per_logging,
        learning_rate=args.lr,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        masking=args.masking,
        missing_k=args.missing_k
    )


if __name__ == "__main__":
    main()

