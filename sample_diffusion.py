import argparse
import os
import json
import torch
import numpy as np
from synthetic_log_gen.models import LogDiffusionModel
from synthetic_log_gen.data.dataset import ALL_CHANNELS

def get_vocab_sizes(vocab_dir, args):
    """
    Load vocab sizes from json files or arguments.
    """
    sizes = {}
    
    # Event ID: Load from vocab.json
    try:
        with open(os.path.join(vocab_dir, "vocab.json")) as f:
            sizes["event"] = len(json.load(f))
    except Exception as e:
        print(f"[WARN] Could not load vocab.json from {vocab_dir}: {e}. Using default 384.")
        sizes["event"] = 384
    
    # CPU, TID, FD from args
    sizes["cpu"] = args.num_cpus
    sizes["tid"] = args.tid_buckets
    sizes["fd"] = args.fd_cap
    
    try:
        with open(os.path.join(vocab_dir, "vocab_comm.json")) as f:
            sizes["comm"] = len(json.load(f))
        with open(os.path.join(vocab_dir, "vocab_ret.json")) as f:
            sizes["ret"] = len(json.load(f))
    except:
        sizes["comm"] = 100
        sizes["ret"] = 1050
    return sizes

def main():
    parser = argparse.ArgumentParser()
    
    # Checkpoint
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out", required=True, help="Output .npz path")
    
    # Model Config (Must match training!)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--vocab-dir", default="dataset/metadata_all_events")
    
    # Vocab/Dim Args (New)
    parser.add_argument("--num-cpus", type=int, default=4, help="Number of CPU cores")
    parser.add_argument("--tid-buckets", type=int, default=256, help="Number of TID hash buckets")
    parser.add_argument("--fd-cap", type=int, default=1025, help="Cap for File Descriptors")
    
    # Generation Config
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Fast Sampling (DDIM)
    parser.add_argument("--use-ddim", action="store_true", help="Use DDIM fast sampling (10-20x faster)")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Number of DDIM steps (default 50 vs 1000 for DDPM)")
    parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic, 1=DDPM)")

    
    args = parser.parse_args()
    
    # 1. Init Model
    vocab_sizes = get_vocab_sizes(args.vocab_dir, args)
    print(f"[Info] Vocab sizes: {vocab_sizes}")
    
    model = LogDiffusionModel(
        vocab_sizes=vocab_sizes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_timesteps=args.steps,
    ).to(args.device)
    
    # 2. Load Checkpoint
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        
    print(f"[Info] Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    
    # 3. Generate Loop
    all_outputs = {}
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    if args.use_ddim:
        print(f"[Generate] Using DDIM fast sampling with {args.ddim_steps} steps (vs {args.steps} for DDPM)")
        print(f"[Generate] Generating {args.num_samples} samples in {num_batches} batches...")
    else:
        print(f"[Generate] Using DDPM sampling with {args.steps} steps")
        print(f"[Generate] Generating {args.num_samples} samples in {num_batches} batches...")
        print(f"[Tip] Use --use-ddim for 10-20x faster generation!")
    
    for i in range(num_batches):
        # Handle last batch size
        curr_bs = min(args.batch_size, args.num_samples - i*args.batch_size)
        
        with torch.no_grad():
            # Use DDIM or DDPM sampling
            if args.use_ddim:
                batch_out = model.sample_ddim(curr_bs, args.seq_len, args.device, 
                                             ddim_steps=args.ddim_steps, eta=args.ddim_eta)
            else:
                batch_out = model.sample(curr_bs, args.seq_len, args.device)
            
        # Accumulate
        for k, v in batch_out.items():
            arr = v.cpu().numpy() # [B, L]
            if k not in all_outputs:
                all_outputs[k] = []
            all_outputs[k].append(arr)
            
        print(f"Batch {i+1}/{num_batches} done ({(i+1)*100//num_batches}%).", end="\r")
        
    print("\n[Info] Concatenating outputs...")
    final_dict = {}
    for k, v_list in all_outputs.items():
        final_dict[k] = np.concatenate(v_list, axis=0)
        
    # 4. Save
    print(f"[Info] Saving to {args.out}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **final_dict)
    print("[Success] Done.")

if __name__ == "__main__":
    main()


# python sample_diffusion.py --ckpt logs_tensorboard/h100_diff_scimark2_256/ckpt_epoch_1.pt --out generated_traces/test_sample.npz --d-model 512 --num-layers 8 --steps 1000 --num-samples 4 --batch-size 4 --device cpu