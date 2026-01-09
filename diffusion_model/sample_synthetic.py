import os
import argparse
import numpy as np
import torch

from dataset.trace_embedding import TraceEmbedding
from diffusion_model.diffusion_denoiser import TransformerDenoiser
from diffusion_model.ddpm_train import DDPM
from diffusion_model.heads import TokenHeads


@torch.no_grad()
def ddpm_sample(denoiser, diffusion, shape, device):
    """Reverse diffusion: x_T -> x_0 (embedding space)."""
    x = torch.randn(shape, device=device)
    for ti in range(diffusion.T - 1, -1, -1):
        t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
        x = diffusion.p_sample(denoiser, x, t)
    return x


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:  # dirname("") happens if user passes just "file.npz"
        os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--batch_samples", type=int, default=512)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--num_events", type=int, default=301)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)

    # Build modules exactly as trained
    embed = TraceEmbedding(
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
        d_model=args.d_model,
        dropout=0.0,
    ).to(device)

    denoiser = TransformerDenoiser(
        d_model=args.d_model,
        nhead=8,
        num_layers=6,
        dropout=0.0
    ).to(device)

    heads = TokenHeads(
        d_model=args.d_model,
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
    ).to(device)

    # Load weights
    embed.load_state_dict(ckpt["embed"])
    denoiser.load_state_dict(ckpt["denoiser"])
    heads.load_state_dict(ckpt["heads"])

    embed.eval()
    denoiser.eval()
    heads.eval()

    diffusion = DDPM(T=args.T, device=device)

    B = int(args.num_samples)
    L = int(args.seq_len)
    bs = max(1, int(args.batch_samples))

    # Pre-allocate outputs on CPU
    out_event = np.empty((B, L), dtype=np.int32)
    out_dt    = np.empty((B, L), dtype=np.uint8)
    out_cpu   = np.empty((B, L), dtype=np.uint8)

    print(f"[INFO] Sampling {B} windows (L={L}) in batches of {bs} ...")

    start = 0
    batch_idx = 0
    while start < B:
        end = min(B, start + bs)
        b = end - start

        # 1) reverse diffuse in embedding space
        x0_hat = ddpm_sample(denoiser, diffusion, (b, L, args.d_model), device)

        # 2) decode to logits
        logits = heads(x0_hat)

        # --- Robust key checks ---
        for k in ("event", "dt", "cpu"):
            if k not in logits:
                raise KeyError(f"heads() did not return key '{k}'. Got keys: {list(logits.keys())}")

        # 3) argmax -> token IDs
        ev = torch.argmax(logits["event"], dim=-1)  # [b,L]
        dt = torch.argmax(logits["dt"],    dim=-1)  # [b,L]
        cp = torch.argmax(logits["cpu"],   dim=-1)  # [b,L]

        # 4) move to CPU + numpy
        ev_np = ev.to("cpu").numpy().astype(np.int32)
        dt_np = dt.to("cpu").numpy().astype(np.uint8)
        cp_np = cp.to("cpu").numpy().astype(np.uint8)

        # --- Hard sanity checks BEFORE saving ---
        # (If any fail, stop immediately so you don’t generate a giant bad file.)
        ev_max = int(ev_np.max()); ev_min = int(ev_np.min())
        dt_max = int(dt_np.max()); dt_min = int(dt_np.min())
        cp_max = int(cp_np.max()); cp_min = int(cp_np.min())

        if not (0 <= ev_min and ev_max < args.num_events):
            raise ValueError(f"[BAD EVENT] min={ev_min} max={ev_max} expected [0,{args.num_events-1}]")
        if not (0 <= dt_min and dt_max < args.num_dt_buckets):
            raise ValueError(f"[BAD DT] min={dt_min} max={dt_max} expected [0,{args.num_dt_buckets-1}]")
        if not (0 <= cp_min and cp_max < args.num_cpus):
            raise ValueError(f"[BAD CPU] min={cp_min} max={cp_max} expected [0,{args.num_cpus-1}]")

        # 5) write into preallocated arrays
        out_event[start:end] = ev_np
        out_dt[start:end]    = dt_np
        out_cpu[start:end]   = cp_np

        # cleanup
        del x0_hat, logits, ev, dt, cp
        if device == "cuda":
            torch.cuda.empty_cache()

        batch_idx += 1
        if batch_idx % 10 == 0 or end == B:
            print(f"[INFO] done {end}/{B}")

        start = end

    ensure_dir(args.out)
    np.savez_compressed(args.out, event=out_event, dt=out_dt, cpu=out_cpu)
    print("[WRITE]", args.out, out_event.shape)

    # Print final ranges
    print("[RANGE] event:", int(out_event.min()), int(out_event.max()))
    print("[RANGE] dt   :", int(out_dt.min()), int(out_dt.max()))
    print("[RANGE] cpu  :", int(out_cpu.min()), int(out_cpu.max()))


if __name__ == "__main__":
    main()