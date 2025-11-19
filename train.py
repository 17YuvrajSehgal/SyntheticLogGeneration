"""
Training script for SSSDS4 model on kernel trace generation.
Based on the ICSE 2025 paper methodology.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from tqdm import tqdm
import json

from data_loader import get_dataloader
from models.sssds4 import SSSDS4, DiffusionScheduler


def train_epoch(model, dataloader, scheduler, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = batch.to(device)
        batch_size, seq_len = batch.shape
        
        # Add channel dimension
        x = batch.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Sample random timesteps
        timesteps = scheduler.sample_timesteps(batch_size, device)
        
        # Add noise to data
        noisy_x, noise = scheduler.add_noise(x, timesteps)
        
        # Predict noise
        predicted_noise = model(noisy_x, timesteps)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(model, dataloader, scheduler, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)
            batch_size, seq_len = batch.shape
            
            x = batch.unsqueeze(-1)
            timesteps = scheduler.sample_timesteps(batch_size, device)
            noisy_x, noise = scheduler.add_noise(x, timesteps)
            predicted_noise = model(noisy_x, timesteps)
            
            loss = nn.functional.mse_loss(predicted_noise, noise)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train SSSDS4 model for kernel trace generation')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., compress-gzip)')
    parser.add_argument('--sequence_length', type=int, default=200,
                       help='Sequence length (default: 200)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of S4 layers (default: 4)')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    train_path = f"Datasets/{args.dataset}/sequence_length_{args.sequence_length}/training"
    test_path = f"Datasets/{args.dataset}/sequence_length_{args.sequence_length}/testing"
    
    print(f"Loading training data from {train_path}")
    train_loader = get_dataloader(
        train_path, 
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        shuffle=True
    )
    
    print(f"Loading test data from {test_path}")
    test_loader = get_dataloader(
        test_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        shuffle=False
    )
    
    # Create model
    model = SSSDS4(
        input_dim=1,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps
    ).to(device)
    
    # Create scheduler
    scheduler = DiffusionScheduler(num_timesteps=args.num_timesteps)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, scheduler, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, test_loader, scheduler, device)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Get normalization values from dataset
        normalization = {
            'min_val': float(train_loader.dataset.min_val),
            'max_val': float(train_loader.dataset.max_val)
        }
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'args': vars(args),
            'normalization': normalization
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest.pt'))
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pt'))
            print(f"Saved best model with val loss: {val_loss:.6f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

