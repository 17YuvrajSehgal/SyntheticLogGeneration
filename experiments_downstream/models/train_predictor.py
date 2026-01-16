"""
Training script for next-event prediction model.

Usage:
    python train_predictor.py --train-data data/real_train.npz --test-data data/real_test.npz --run-name real_baseline
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import sys

from next_event_predictor import NextEventPredictor, NextEventPredictorEventOnly

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EventSequenceDataset(Dataset):
    """
    Dataset for next-event prediction.
    Creates sliding windows from traces.
    """
    
    def __init__(self, npz_path, seq_len=128, stride=64, channels=None):
        """
        Args:
            npz_path: Path to .npz file with traces
            seq_len: Length of input sequence
            stride: Stride for sliding window
            channels: List of channels to use (default: all)
        """
        self.seq_len = seq_len
        self.stride = stride
        
        # Load data
        data = np.load(npz_path)
        
        # Default channels
        if channels is None:
            channels = ['event', 'dt', 'cpu', 'tid', 'comm', 'ret']
        
        self.channels = channels
        self.data = {ch: data[ch] for ch in channels if ch in data}
        
        # Create sliding windows
        self.windows = []
        num_traces, trace_len = self.data['event'].shape
        
        for i in range(num_traces):
            for start in range(0, max(1, trace_len - seq_len), stride):
                end = start + seq_len
                # Ensure we have a target event after the sequence
                if end < trace_len:
                    self.windows.append((i, start, end))
        
        print(f"[Dataset] Loaded {len(self.windows)} windows from {num_traces} traces")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        trace_idx, start, end = self.windows[idx]
        
        # Input: [start:end]
        inputs = {}
        for ch in self.channels:
            inputs[ch] = torch.tensor(self.data[ch][trace_idx, start:end], dtype=torch.long)
        
        # Target: next event after sequence
        target = torch.tensor(self.data['event'][trace_idx, end], dtype=torch.long)
        
        return inputs, target


def get_vocab_sizes(vocab_dir):
    """Load vocabulary sizes from metadata."""
    sizes = {}
    
    # Event
    with open(os.path.join(vocab_dir, 'vocab.json')) as f:
        sizes['event'] = len(json.load(f))
    
    # Comm, Ret
    with open(os.path.join(vocab_dir, 'vocab_comm.json')) as f:
        sizes['comm'] = len(json.load(f))
    with open(os.path.join(vocab_dir, 'vocab_ret.json')) as f:
        sizes['ret'] = len(json.load(f))
    
    # Fixed sizes
    sizes['cpu'] = 4
    sizes['tid'] = 256
    sizes['fd'] = 1025
    
    return sizes


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Predictions
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    for inputs, targets in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)
        
        # Forward
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        total_loss += loss.item()
        
        # Predictions
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Top-K accuracy
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    top5_acc = np.mean([target in np.argsort(probs)[-5:] for target, probs in zip(all_targets, all_probs)])
    top10_acc = np.mean([target in np.argsort(probs)[-10:] for target, probs in zip(all_targets, all_probs)])
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'top5_accuracy': top5_acc,
        'top10_accuracy': top10_acc,
    }
    
    return metrics, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--train-data', required=True, help='Path to training .npz')
    parser.add_argument('--test-data', required=True, help='Path to test .npz')
    parser.add_argument('--vocab-dir', default='dataset/metadata_all_events')
    
    # Model
    parser.add_argument('--model-type', default='full', choices=['full', 'event_only'])
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    
    # Output
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output-dir', default='experiments_downstream/results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"[Info] Run: {args.run_name}")
    print(f"[Info] Device: {args.device}")
    
    # Load vocab sizes
    vocab_sizes = get_vocab_sizes(args.vocab_dir)
    print(f"[Info] Vocab sizes: {vocab_sizes}")
    
    # Create datasets
    channels = ['event', 'dt', 'cpu', 'tid', 'comm', 'ret'] if args.model_type == 'full' else ['event']
    
    train_dataset = EventSequenceDataset(
        args.train_data,
        seq_len=args.seq_len,
        stride=args.stride,
        channels=channels
    )
    
    test_dataset = EventSequenceDataset(
        args.test_data,
        seq_len=args.seq_len,
        stride=args.stride,
        channels=channels
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    if args.model_type == 'full':
        model = NextEventPredictor(
            vocab_sizes=vocab_sizes,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_len=args.seq_len
        )
    else:
        model = NextEventPredictorEventOnly(
            num_events=vocab_sizes['event'],
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_len=args.seq_len
        )
    
    model = model.to(args.device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        
        # Evaluate
        test_metrics, test_preds, test_targets = evaluate(
            model, test_loader, criterion, args.device, vocab_sizes['event']
        )
        
        # Log
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Acc: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"Test Top-5 Acc: {test_metrics['top5_accuracy']:.4f}")
        print(f"Test Top-10 Acc: {test_metrics['top10_accuracy']:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            **{f'test_{k}': v for k, v in test_metrics.items()}
        })
        
        # Save best model
        if test_metrics['f1_macro'] > best_f1:
            best_f1 = test_metrics['f1_macro']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': test_metrics,
            }, os.path.join(run_dir, 'best_model.pt'))
            
            print(f"[Saved] New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f"[Early Stop] No improvement for {args.patience} epochs")
            break
    
    # Save final results
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model and evaluate
    checkpoint = torch.load(os.path.join(run_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics, final_preds, final_targets = evaluate(
        model, test_loader, criterion, args.device, vocab_sizes['event']
    )
    
    # Save final metrics
    with open(os.path.join(run_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save predictions
    np.savez(
        os.path.join(run_dir, 'predictions.npz'),
        predictions=final_preds,
        targets=final_targets
    )
    
    print(f"\n[Done] Best F1 (macro): {best_f1:.4f}")
    print(f"[Saved] Results to {run_dir}")


if __name__ == '__main__':
    main()
