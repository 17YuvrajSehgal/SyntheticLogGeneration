import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import IMPROVED model
from synthetic_log_gen.data.dataset import make_dataloaders, SampleConfig, ALL_CHANNELS
from synthetic_log_gen.models.diffusion_better import LogDiffusionModelBetter

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

def compute_real_data_statistics(train_dl, device):
    """
    Compute target repetition count from real training data.
    This will be used as the target for the repetition loss.
    """
    print("[Info] Computing real data statistics for repetition target...")
    
    total_repeats = 0
    total_traces = 0
    
    with torch.no_grad():
        for i, batch in enumerate(train_dl):
            if i >= 100:  # Sample first 100 batches
                break
            
            events = batch['event'].to(device)  # [B, L]
            
            # Count immediate repeats
            repeats = (events[:, :-1] == events[:, 1:]).float().sum(dim=1)  # [B]
            total_repeats += repeats.sum().item()
            total_traces += events.shape[0]
    
    avg_repeats = total_repeats / total_traces if total_traces > 0 else 279
    print(f"[Info] Average repeats per trace in real data: {avg_repeats:.1f}")
    
    return int(avg_repeats)

def main():
    parser = argparse.ArgumentParser(description="Train IMPROVED Synthetic Log Model with Repetition Awareness")
    
    # Data Args
    parser.add_argument("--data-root", required=True, help="Root dir containing train/val/test folders")
    parser.add_argument("--benchmark", default=None, help="Subdirectory name")
    parser.add_argument("--vocab-dir", default="dataset/metadata_all_events", help="Path to vocab jsons")
    parser.add_argument("--channels", nargs="+", default=ALL_CHANNELS, help="Features to use")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    # Vocab/Dim Args
    parser.add_argument("--num-cpus", type=int, default=4, help="Number of CPU cores (dataset specific)")
    parser.add_argument("--tid-buckets", type=int, default=256, help="Number of TID hash buckets")
    parser.add_argument("--fd-cap", type=int, default=1025, help="Cap for File Descriptors")
    
    # Model Args
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1000)
    
    # NEW: Repetition-aware parameters
    parser.add_argument("--target-repeats", type=int, default=None, 
                       help="Target repetition count (auto-computed from data if not specified)")
    parser.add_argument("--repetition-weight", type=float, default=0.05,
                       help="Weight for repetition loss (default: 0.05)")
    parser.add_argument("--transition-weight", type=float, default=0.03,
                       help="Weight for transition frequency loss (default: 0.03)")
    
    # Optimization Args
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Limit steps per epoch")
    parser.add_argument("--mixed-precision", default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    
    # Logging
    parser.add_argument("--log-dir", default="logs_tensorboard")
    parser.add_argument("--run-name", default=None)
    
    args = parser.parse_args()
    
    # Enable TF32
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    
    # Setup Logging
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f"diffusion_better_{args.benchmark}_{args.seq_len}_{timestamp}"
    
    log_path = os.path.join(args.log_dir, args.run_name)
    writer = SummaryWriter(log_path)
    print(f"[Experiment] Logging to {log_path}")
    print(f"[Experiment] IMPROVED MODEL with Repetition Awareness")
    print(f"[Experiment] Repetition weight: {args.repetition_weight}, Transition weight: {args.transition_weight}")

    # 1. Setup Data
    cfg = SampleConfig(
        seq_len=args.seq_len,
        channels=tuple(args.channels),
        return_dict=True
    )
    
    train_dl, val_dl, test_dl = make_dataloaders(
        root_dir=args.data_root,
        benchmark=args.benchmark,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        config=cfg,
        prefetch_factor=4,
        cache_shards=8
    )
    
    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_sizes = get_vocab_sizes(args.vocab_dir, args)
    print(f"[Info] Vocab Sizes: {vocab_sizes}")
    
    # Compute target repeats from real data if not specified
    if args.target_repeats is None:
        args.target_repeats = compute_real_data_statistics(train_dl, device)
    
    print(f"[Info] Target repetition count: {args.target_repeats}")
    
    # Create IMPROVED model
    model = LogDiffusionModelBetter(
        vocab_sizes=vocab_sizes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_timesteps=args.steps,
        dropout=args.dropout,
        target_repeats=args.target_repeats,
        repetition_weight=args.repetition_weight,
        transition_weight=args.transition_weight
    ).to(device)

    # Torch.compile
    if args.compile:
        try:
            print("[Info] Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}. Running eager mode.")

    # DataParallel
    if torch.cuda.device_count() > 1 and not args.compile:
        print(f"[Info] Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = nn.DataParallel(model)

    # 3. Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Mixed Precision Setup
    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    mp_dtype = dtype_map[args.mixed_precision]
    use_amp = args.mixed_precision != "no"
    scaler = torch.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    
    # Calculate training statistics
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    
    print(f"\n{'='*60}")
    print("Training Progress Estimation")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_dl.dataset):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total epochs: {args.epochs}")
    print(f"Total steps: {total_steps:,}")
    print(f"\nEstimated time per epoch: ~{steps_per_epoch * 0.5 / 60:.1f} minutes")
    print(f"Estimated total time: ~{total_steps * 0.5 / 3600:.1f} hours")
    print(f"(Assumes ~0.5 seconds per step)")
    print(f"{'='*60}\n")
    
    global_step = 0
    print(f"[Train] Starting {args.epochs} epochs on {device}...")
    
    epoch_loss_sum = 0.0
    epoch_loss_count = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_steps = 0
        for i, batch in enumerate(train_dl):
            if args.max_steps_per_epoch is not None and epoch_steps >= args.max_steps_per_epoch:
                print(f"\n[INFO] Reached max steps per epoch ({args.max_steps_per_epoch}). Ending epoch {epoch} early.")
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda', dtype=mp_dtype, enabled=use_amp):
                loss, metrics = model(batch)
                if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                    loss = loss.mean()
                    # Handle DataParallel outputs - mean all tensor metrics
                    for k, v in metrics.items():
                        if k == "recon_loss_per_channel":
                            # Nested dict - mean each channel loss
                            metrics[k] = {ch: ch_loss.mean() if isinstance(ch_loss, torch.Tensor) and ch_loss.ndim > 0 else ch_loss 
                                         for ch, ch_loss in v.items()}
                        elif isinstance(v, torch.Tensor) and v.ndim > 0:
                            metrics[k] = v.mean()

            scaler.scale(loss).backward()
            
            # Gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss_sum += loss.item()
            epoch_loss_count += 1
            
            # Logging
            if i % 10 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/GradientNorm", total_norm, global_step)
                writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]['lr'], global_step)
                
                # NEW: Log repetition metrics
                if 'repetition_loss' in metrics:
                    writer.add_scalar("Train/repetition_loss", metrics['repetition_loss'].item(), global_step)
                if 'transition_loss' in metrics:
                    writer.add_scalar("Train/transition_loss", metrics['transition_loss'].item(), global_step)
                if 'pred_repeats' in metrics:
                    writer.add_scalar("Train/pred_repeats", metrics['pred_repeats'].item(), global_step)
                if 'target_repeats' in metrics:
                    writer.add_scalar("Train/target_repeats", metrics['target_repeats'].item(), global_step)
                
                # Other metrics
                for k, v in metrics.items():
                    if k == "recon_loss_per_channel":
                        for channel, channel_loss in v.items():
                            val = channel_loss.item() if isinstance(channel_loss, torch.Tensor) else channel_loss
                            writer.add_scalar(f"Train/Recon_{channel}", val, global_step)
                    elif k not in ['repetition_loss', 'transition_loss', 'pred_repeats', 'target_repeats']:
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        writer.add_scalar(f"Train/{k}", val, global_step)
                
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | "
                      f"RepLoss: {metrics.get('repetition_loss', 0):.4f} | "
                      f"PredRep: {metrics.get('pred_repeats', 0):.1f} | "
                      f"TargetRep: {metrics.get('target_repeats', 0):.1f}", end="\r")
            
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
        val_rep_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for val_batch in val_dl:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                
                with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda', dtype=mp_dtype, enabled=use_amp):
                    v_loss, v_metrics = model(val_batch)
                    if isinstance(v_loss, torch.Tensor) and v_loss.ndim > 0:
                        v_loss = v_loss.mean()
                        # Handle DataParallel outputs
                        for k, v in v_metrics.items():
                            if k == "recon_loss_per_channel":
                                v_metrics[k] = {ch: ch_loss.mean() if isinstance(ch_loss, torch.Tensor) and ch_loss.ndim > 0 else ch_loss 
                                               for ch, ch_loss in v.items()}
                            elif isinstance(v, torch.Tensor) and v.ndim > 0:
                                v_metrics[k] = v.mean()
                    
                    val_loss += v_loss.item()
                    val_latent_loss += v_metrics['latent_loss'].item()
                    val_recon_loss += v_metrics['recon_loss'].item()
                    if 'repetition_loss' in v_metrics:
                        val_rep_loss += v_metrics['repetition_loss'].item()
                
                val_steps += 1
                if val_steps >= 50:
                    break
        
        # Log validation metrics
        if val_steps > 0:
            writer.add_scalar("Val/Loss", val_loss / val_steps, epoch)
            writer.add_scalar("Val/LatentLoss", val_latent_loss / val_steps, epoch)
            writer.add_scalar("Val/ReconLoss", val_recon_loss / val_steps, epoch)
            writer.add_scalar("Val/RepetitionLoss", val_rep_loss / val_steps, epoch)
            print(f"[Validation] Loss: {val_loss / val_steps:.4f}, RepLoss: {val_rep_loss / val_steps:.4f}")
        
        print(f"[Epoch {epoch}] Completed.")
        
        # Save checkpoint
        raw_model = model
        if hasattr(raw_model, "module"):
             raw_model = raw_model.module
        if hasattr(raw_model, "_orig_mod"):
             raw_model = raw_model._orig_mod
             
        ckpt_path = os.path.join(log_path, f"ckpt_epoch_{epoch}.pt")
        torch.save(raw_model.state_dict(), ckpt_path)
        
    writer.close()
    print("\n[Success] Training Complete with IMPROVED model!")
    print(f"[Info] Checkpoints saved to: {log_path}")

if __name__ == "__main__":
    main()
