# sample_autoregressive.py
import os
import argparse
import numpy as np
import torch

from dataset_processing.trace_embedding import TraceEmbedding
from diffusion_model.heads import TokenHeads
from autoregressive_transformer.autoregressive_transformer import ARTransformer


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


@torch.no_grad()
def sample_ar(ar_model, heads, num_samples, seq_len, device,
              num_events, num_dt_buckets, num_cpus,
              temperature=1.0, greedy=True):
    """
    Generates [B, L, 3] token ids autoregressively.
    Uses a simple BOS = (0,0,0) as the first token.
    """
    B = num_samples
    L = seq_len

    # output tokens
    x = torch.zeros((B, L, 3), dtype=torch.long, device=device)  # BOS-filled

    for t in range(0, L - 1):
        # run model on prefix length t+1
        h = ar_model(x[:, :t+1, :])        # [B, t+1, D]
        logits = heads(h)                  # dict [B, t+1, V]

        # predict next token at position t (which corresponds to next index t+1)
        ev_logits = logits["event"][:, -1, :] / max(1e-8, temperature)
        dt_logits = logits["dt"][:, -1, :] / max(1e-8, temperature)
        cp_logits = logits["cpu"][:, -1, :] / max(1e-8, temperature)

        if greedy:
            ev = torch.argmax(ev_logits, dim=-1)
            dt = torch.argmax(dt_logits, dim=-1)
            cp = torch.argmax(cp_logits, dim=-1)
        else:
            ev = torch.distributions.Categorical(logits=ev_logits).sample()
            dt = torch.distributions.Categorical(logits=dt_logits).sample()
            cp = torch.distributions.Categorical(logits=cp_logits).sample()

        x[:, t+1, 0] = ev
        x[:, t+1, 1] = dt
        x[:, t+1, 2] = cp

    # hard range checks
    ev_min, ev_max = int(x[..., 0].min()), int(x[..., 0].max())
    dt_min, dt_max = int(x[..., 1].min()), int(x[..., 1].max())
    cp_min, cp_max = int(x[..., 2].min()), int(x[..., 2].max())
    if not (0 <= ev_min and ev_max < num_events):
        raise ValueError(f"[BAD EVENT] min={ev_min} max={ev_max} expected [0,{num_events-1}]")
    if not (0 <= dt_min and dt_max < num_dt_buckets):
        raise ValueError(f"[BAD DT] min={dt_min} max={dt_max} expected [0,{num_dt_buckets-1}]")
    if not (0 <= cp_min and cp_max < num_cpus):
        raise ValueError(f"[BAD CPU] min={cp_min} max={cp_max} expected [0,{num_cpus-1}]")

    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=200)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)

    ap.add_argument("--num_events", type=int, default=301)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)

    embed = TraceEmbedding(
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
        d_model=args.d_model,
        dropout=0.0,
    ).to(device)

    ar_model = ARTransformer(
        embed_module=embed,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=0.0,
    ).to(device)

    heads = TokenHeads(
        d_model=args.d_model,
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
    ).to(device)

    embed.load_state_dict(ckpt["embed"])
    ar_model.load_state_dict(ckpt["ar_model"])
    heads.load_state_dict(ckpt["heads"])

    embed.eval(); ar_model.eval(); heads.eval()

    x = sample_ar(
        ar_model, heads,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        device=device,
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
        temperature=args.temperature,
        greedy=args.greedy,
    )

    # write npz like diffusion
    x_cpu = x.to("cpu").numpy()
    out_event = x_cpu[..., 0].astype(np.int32)
    out_dt    = x_cpu[..., 1].astype(np.uint8)
    out_cpu   = x_cpu[..., 2].astype(np.uint8)

    ensure_dir(args.out)
    np.savez_compressed(args.out, event=out_event, dt=out_dt, cpu=out_cpu)
    print("[WRITE]", args.out, out_event.shape)
    print("[RANGE] event:", int(out_event.min()), int(out_event.max()))
    print("[RANGE] dt   :", int(out_dt.min()), int(out_dt.max()))
    print("[RANGE] cpu  :", int(out_cpu.min()), int(out_cpu.max()))


if __name__ == "__main__":
    main()
