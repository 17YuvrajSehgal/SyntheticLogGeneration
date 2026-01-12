import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# New Package Imports
from synthetic_log_gen.data.dataset import make_dataloaders, SampleConfig, ALL_CHANNELS
from synthetic_log_gen.models import LogDiffusionModel

def get_vocab_sizes(vocab_dir):
    """
    Load vocab sizes from json files or hardcoded logic.
    """
    sizes = {}

    # CPU (0-3)
    sizes["cpu"] = 4
    
    # TID (Hash) -> 256
    sizes["tid"] = 256
    
    # FD (Cap) -> 1025
    sizes["fd"] = 1025
    
    try:
        with open(os.path.join(vocab_dir, "vocab.json")) as f:
            sizes["event"] = len(json.load(f))

        with open(os.path.join(vocab_dir, "vocab_comm.json")) as f:
            sizes["comm"] = len(json.load(f))
        
        with open(os.path.join(vocab_dir, "vocab_ret.json")) as f:
            sizes["ret"] = len(json.load(f))
    except Exception as e:
        print(f"[WARN] Could not load vocabs from {vocab_dir}: {e}. Using defaults.")
        sizes["event"] = 384
        sizes["comm"] = 100
        sizes["ret"] = 1050

    return sizes

def main():
    parser = argparse.ArgumentParser(description="Train Synthetic Log Model")
    
    # Data Args
    parser.add_argument("--data-root", required=True, help="Root dir containing train/val/test folders")
    parser.add_argument("--benchmark", default=None, help="Subdirectory name")
    parser.add_argument("--vocab-dir", default="dataset/metadata_all_events", help="Path to vocab jsons")
    parser.add_argument("--channels", nargs="+", default=ALL_CHANNELS, help="Features to use")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model Args
    parser.add_argument("--model-type", default="diffusion", choices=["dummy", "diffusion"])
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1000)
    
    # Optimization Args (H100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mixed-precision", default="no", choices=["no", "fp16", "bf16"], help="Use mixed precision (bf16 for H100)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() (PyTorch 2.0)")
    
    # Logging
    parser.add_argument("--log-dir", default="logs_tensorboard")
    parser.add_argument("--run-name", default=None)
    
    args = parser.parse_args()
    
    # --- Optimization Setup (TF32) ---
    # Enable TF32 for Ampere/Hopper GPUs (significant speedup)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup Logging
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prec = args.mixed_precision
        comp = "compiled" if args.compile else "raw"
        args.run_name = f"{args.model_type}_{prec}_{comp}_{timestamp}"
    
    log_path = os.path.join(args.log_dir, args.run_name)
    writer = SummaryWriter(log_path)
    print(f"[Experiment] Logging to {log_path}")
    print(f"[Experiment] Optimizations: Mixed Precision={args.mixed_precision}, Compile={args.compile}, TF32=True")

    # 1. Setup Data
    cfg = SampleConfig(
        seq_len=args.seq_len,
        channels=tuple(args.channels),
        return_dict=True # Model expects dictionary
    )
    
    train_dl, val_dl, test_dl = make_dataloaders(
        root_dir=args.data_root,
        benchmark=args.benchmark,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        config=cfg
    )
    
    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_sizes = get_vocab_sizes(args.vocab_dir)
    print(f"[Info] Vocab Sizes: {vocab_sizes}")
    
    if args.model_type == "diffusion":
        model = LogDiffusionModel(
            vocab_sizes=vocab_sizes,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_timesteps=args.steps,
            dropout=args.dropout
        ).to(device)
    else:
        # Dummy fallback
        model = nn.Linear(1, 1).to(device) 

    # --- Torch.compile ---
    if args.compile:
        try:
            print("[Info] Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}. Running eager mode.")

    # --- DataParallel ---
    # Note: torch.compile and DataParallel generally don't mix well in current PT versions
    # Recommended: DDP. But for single-node multi-GPU, DP is okay if not compiled.
    # If compiled, use DDP usually. For simplicity here, we prioritize compile if 1 GPU, DP if >1GPU?
    # H100 usually 1 big GPU job.
    if torch.cuda.device_count() > 1 and not args.compile:
        print(f"[Info] Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = nn.DataParallel(model)

    # 3. Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Mixed Precision Setup
    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    mp_dtype = dtype_map[args.mixed_precision]
    use_amp = args.mixed_precision != "no"
    scaler = torch.amp.GradScaler(enabled=(args.mixed_precision == "fp16")) # Scaler mostly needed for fp16, bf16 doesn't need it but harmless
    
    global_step = 0
    print(f"[Train] Starting {args.epochs} epochs on {device}...")
    
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(train_dl):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Autocast Context
            with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda', dtype=mp_dtype, enabled=use_amp):
                if args.model_type == "diffusion":
                    loss, metrics = model(batch)
                    # If DataParallel, loss is a vector. Mean it.
                    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                        loss = loss.mean()
                        metrics = {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
                else:
                    loss = torch.tensor(0.0, requires_grad=True, device=device)
                    metrics = {}

            # Backward with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            if i % 10 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                for k, v in metrics.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    writer.add_scalar(f"Train/{k}", val, global_step)
                
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}", end="\r")
            
            global_step += 1
            
        # Validation Loop (Optional: Implement sampling here)
        print(f"\n[Epoch {epoch}] Completed.")
        
        # Save checkpoint (Unwrap from compile/DataParallel)
        raw_model = model
        if hasattr(raw_model, "module"): # DataParallel
             raw_model = raw_model.module
        if hasattr(raw_model, "_orig_mod"): # torch.compile
             raw_model = raw_model._orig_mod
             
        ckpt_path = os.path.join(log_path, f"ckpt_epoch_{epoch}.pt")
        torch.save(raw_model.state_dict(), ckpt_path)
        
    writer.close()
    print("\n[Success] Training Complete.")

if __name__ == "__main__":
    main()
