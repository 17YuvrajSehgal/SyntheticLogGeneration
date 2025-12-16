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
    x = torch.randn(shape, device=device)
    for ti in range(diffusion.T - 1, -1, -1):
        t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
        x = diffusion.p_sample(model, x, t)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--batch_samples", type=int, default=512, help="how many samples to generate per GPU batch")
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
    denoiser = TransformerDenoiser(d_model=args.d_model, nhead=8, num_layers=6, dropout=0.0).to(device)
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
    L = args.seq_len

    # Pre-allocate outputs on CPU (100k x 200 is totally fine)
    out_event = np.empty((B, L), dtype=np.int32)
    out_dt    = np.empty((B, L), dtype=np.uint8)
    out_cpu   = np.empty((B, L), dtype=np.uint8)

    bs = max(1, int(args.batch_samples))
    print(f"[INFO] Sampling {B} windows in batches of {bs}...")

    start = 0
    while start < B:
        end = min(B, start + bs)
        b = end - start

        x0_hat = sample(denoiser, diffusion, (b, L, args.d_model), device)
        logits = heads(x0_hat)

        out_event[start:end] = torch.argmax(logits["event"], dim=-1).to("cpu", non_blocking=True).numpy().astype(np.int32)
        out_dt[start:end]    = torch.argmax(logits["dt"],    dim=-1).to("cpu", non_blocking=True).numpy().astype(np.uint8)
        out_cpu[start:end]   = torch.argmax(logits["cpu"],   dim=-1).to("cpu", non_blocking=True).numpy().astype(np.uint8)

        # Free GPU tensors between batches
        del x0_hat, logits
        if device == "cuda":
            torch.cuda.empty_cache()

        if (start // bs) % 10 == 0:
            print(f"[INFO] done {end}/{B}")

        start = end

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, event=out_event, dt=out_dt, cpu=out_cpu)
    print("[WRITE]", args.out, out_event.shape)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
