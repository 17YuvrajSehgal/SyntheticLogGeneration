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

def get_vocab_sizes(vocab_dir, args):
    """
    Load vocab sizes from json files or arguments.
    """
    sizes = {}

    # CPU, TID, FD from args
    sizes["cpu"] = args.num_cpus
    sizes["tid"] = args.tid_buckets
    sizes["fd"] = args.fd_cap
    
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

    # Vocab/Dim Args (New)
    parser.add_argument("--num-cpus", type=int, default=4, help="Number of CPU cores (dataset specific)")
    parser.add_argument("--tid-buckets", type=int, default=256, help="Number of TID hash buckets")
    parser.add_argument("--fd-cap", type=int, default=1025, help="Cap for File Descriptors")
    
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
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Limit steps per epoch (useful for large datasets)")
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
    vocab_sizes = get_vocab_sizes(args.vocab_dir, args)
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
    
    # Track epoch-level statistics
    epoch_loss_sum = 0.0
    epoch_loss_count = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_steps = 0
        for i, batch in enumerate(train_dl):
            # Check if we've reached max steps for this epoch
            if args.max_steps_per_epoch is not None and epoch_steps >= args.max_steps_per_epoch:
                print(f"\n[INFO] Reached max steps per epoch ({args.max_steps_per_epoch}). Ending epoch {epoch} early.")
                break
            
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
            
            # Compute gradient norm (before optimizer step)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            scaler.step(optimizer)
            scaler.update()
            
            # Track epoch statistics
            epoch_loss_sum += loss.item()
            epoch_loss_count += 1
            
            # Logging
            if i % 10 == 0:
                # Basic losses
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/GradientNorm", total_norm, global_step)
                writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]['lr'], global_step)
                
                # Metrics (latent_loss, recon_loss)
                for k, v in metrics.items():
                    if k == "recon_loss_per_channel":
                        # Log per-channel reconstruction losses
                        for channel, channel_loss in v.items():
                            val = channel_loss.item() if isinstance(channel_loss, torch.Tensor) else channel_loss
                            writer.add_scalar(f"Train/Recon_{channel}", val, global_step)
                    else:
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        writer.add_scalar(f"Train/{k}", val, global_step)
                
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | GradNorm: {total_norm:.4f}", end="\r")
            
            global_step += 1
            epoch_steps += 1
            
        # Epoch-level statistics
        avg_epoch_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0
        writer.add_scalar("Epoch/AvgTrainLoss", avg_epoch_loss, epoch)
        writer.add_scalar("Epoch/StepsCompleted", epoch_steps, epoch)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        
        # Validation Loop
        print(f"\n[Epoch {epoch}] Running validation...")
        model.eval()
        val_loss = 0.0
        val_latent_loss = 0.0
        val_recon_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for val_batch in val_dl:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                
                with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda', dtype=mp_dtype, enabled=use_amp):
                    if args.model_type == "diffusion":
                        v_loss, v_metrics = model(val_batch)
                        if isinstance(v_loss, torch.Tensor) and v_loss.ndim > 0:
                            v_loss = v_loss.mean()
                            v_metrics = {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in v_metrics.items()}
                        
                        val_loss += v_loss.item()
                        val_latent_loss += v_metrics['latent_loss'].item()
                        val_recon_loss += v_metrics['recon_loss'].item()
                
                val_steps += 1
                if val_steps >= 50:  # Limit validation to 50 batches for speed
                    break
        
        # Log validation metrics
        if val_steps > 0:
            writer.add_scalar("Val/Loss", val_loss / val_steps, epoch)
            writer.add_scalar("Val/LatentLoss", val_latent_loss / val_steps, epoch)
            writer.add_scalar("Val/ReconLoss", val_recon_loss / val_steps, epoch)
            print(f"[Validation] Loss: {val_loss / val_steps:.4f}")
        
        # Log parameter histograms every 10 epochs
        if epoch % 10 == 0:
            raw_model_for_hist = model
            if hasattr(raw_model_for_hist, "module"):
                raw_model_for_hist = raw_model_for_hist.module
            if hasattr(raw_model_for_hist, "_orig_mod"):
                raw_model_for_hist = raw_model_for_hist._orig_mod
            
            for name, param in raw_model_for_hist.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f"Params/{name}", param.data, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"Gradients/{name}", param.grad.data, epoch)
        
        # Generate samples every 10 epochs (after epoch 0)
        if epoch % 10 == 0 and epoch > 0:
            print(f"[Epoch {epoch}] Generating samples...")
            raw_model_for_sample = model
            if hasattr(raw_model_for_sample, "module"):
                raw_model_for_sample = raw_model_for_sample.module
            if hasattr(raw_model_for_sample, "_orig_mod"):
                raw_model_for_sample = raw_model_for_sample._orig_mod
            
            with torch.no_grad():
                samples = raw_model_for_sample.sample(batch_size=4, seq_len=args.seq_len, device=device)
                
                # Log sample statistics
                for channel, data in samples.items():
                    if channel != 'dt':  # Discrete channels
                        # Flatten and create histogram of values
                        flat_data = data.flatten().cpu()
                        writer.add_histogram(f"Samples/{channel}_distribution", flat_data.float(), epoch)
        
        print(f"[Epoch {epoch}] Completed.")
        
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
