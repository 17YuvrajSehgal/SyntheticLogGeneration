import os
import argparse
import numpy as np
import torch

from dataset_processing.trace_embedding import TraceEmbedding
from diffusion_model.diffusion_denoiser import TransformerDenoiser
from diffusion_model.ddpm_train import DDPM
from diffusion_model.heads import TokenHeads


@torch.no_grad()
def sample(model, diffusion, shape, device):
    """
    shape: (B, L, D)
    """
    x = torch.randn(shape, device=device)
    for ti in range(diffusion.T - 1, -1, -1):
        t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
        x = diffusion.p_sample(model, x, t)
    return x  # approx x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)  # output .npz path
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--T", type=int, default=1000)

    ap.add_argument("--num_events", type=int, default=301)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)

    embed = TraceEmbedding(
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
        d_model=args.d_model,
        dropout=0.0,
    ).to(device)
    denoiser = TransformerDenoiser(
        d_model=args.d_model, nhead=8, num_layers=6, dropout=0.0
    ).to(device)
    heads = TokenHeads(
        d_model=args.d_model,
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
    ).to(device)

    embed.load_state_dict(ckpt["embed"])
    denoiser.load_state_dict(ckpt["denoiser"])
    heads.load_state_dict(ckpt["heads"])

    embed.eval(); denoiser.eval(); heads.eval()

    diffusion = DDPM(T=args.T, device=device)

    B = args.num_samples
    x0_hat = sample(denoiser, diffusion, (B, args.seq_len, args.d_model), device)

    logits = heads(x0_hat)
    event = torch.argmax(logits["event"], dim=-1).cpu().numpy().astype(np.int32)  # [B,L]
    dt    = torch.argmax(logits["dt"], dim=-1).cpu().numpy().astype(np.uint8)    # [B,L]
    cpu   = torch.argmax(logits["cpu"], dim=-1).cpu().numpy().astype(np.uint8)   # [B,L]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, event=event, dt=dt, cpu=cpu)
    print("[WRITE]", args.out, event.shape)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
