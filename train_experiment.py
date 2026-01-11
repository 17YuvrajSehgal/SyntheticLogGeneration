import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from synthetic_log_gen.data.dataset import make_dataloaders, SampleConfig, ALL_CHANNELS

def main():
    parser = argparse.ArgumentParser(description="Train Synthetic Log Model")
    
    # Data Args
    parser.add_argument("--data-root", required=True, help="Root dir containing train/val/test folders")
    parser.add_argument("--benchmark", default=None, help="Subdirectory name")
    parser.add_argument("--channels", nargs="+", default=ALL_CHANNELS, help="Features to use (e.g. event dt cpu)")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model Args
    parser.add_argument("--model-type", default="dummy", choices=["dummy", "diffusion", "lstm", "transformer"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    
    args = parser.parse_args()
    
    print(f"[Experiment] Model: {args.model_type}")
    print(f"             Features: {args.channels}")
    print(f"             Data: {args.data_root}")

    # 1. Setup Data
    cfg = SampleConfig(
        seq_len=args.seq_len,
        channels=tuple(args.channels),
        return_dict=False # Stacked tensor
    )
    
    train_dl, val_dl, test_dl = make_dataloaders(
        root_dir=args.data_root,
        benchmark=args.benchmark,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        config=cfg
    )
    
    print(f"[Data] Train batches: {len(train_dl)}")
    
    # 2. Setup Model (Stub for now)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(args.channels)
    
    if args.model_type == "dummy":
        # simple linear layer to test pipeline
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        ).to(device)
    else:
        raise NotImplementedError(f"Model {args.model_type} not implemented yet.")
        
    print(f"[Model] Created {args.model_type} on {device}")
    
    # 3. Dummy Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    model.train()
    for text_batch in train_dl:
        # batch: [B, L, C]
        x = text_batch.float().to(device) # dummy model needs float
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, x) # Autoencoder task
        loss.backward()
        optimizer.step()
        
        print(f"Batch Loss: {loss.item():.4f}", end="\r")
        break # Just run one batch to prove it works
        
    print("\n[Success] Pipeline verified.")

if __name__ == "__main__":
    main()

# python train_experiment.py --data-root dataset/window_shards --channels event dt cpu tid --model-type dummy