# sample_autoregressive.py
import os
import glob
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


def _load_first_tokens_npz(npz_path: str):
    """
    Returns first tokens [N, 3] from a shard as int64:
      columns: event, dt, cpu
    """
    d = np.load(npz_path)
    ev0 = d["event"][:, 0].astype(np.int64)
    dt0 = d["dt"][:, 0].astype(np.int64)
    cp0 = d["cpu"][:, 0].astype(np.int64)
    return np.stack([ev0, dt0, cp0], axis=1)


def sample_first_tokens_from_real(
    root_shards: str,
    benchmark: str,
    split: str,
    B: int,
    max_shards: int = 10,
    seed: int = 0,
):
    """
    Samples B first-step tokens from real shards (split) to seed x[:,0,:].
    This avoids artificial BOS=(0,0,0) bias.
    """
    rng = np.random.default_rng(seed)
    pattern = os.path.join(root_shards, benchmark, split, "*.npz")
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found for pattern: {pattern}")

    shard_paths = shard_paths[:max_shards]

    first_tokens = []
    for p in shard_paths:
        first_tokens.append(_load_first_tokens_npz(p))

    first_tokens = np.concatenate(first_tokens, axis=0)  # [M, 3]
    idx = rng.integers(0, first_tokens.shape[0], size=B)
    return first_tokens[idx]  # [B, 3]


@torch.no_grad()
def sample_ar(
    ar_model,
    heads,
    num_samples,
    seq_len,
    device,
    num_events,
    num_dt_buckets,
    num_cpus,
    temperature=1.0,
    greedy=True,
    x0: torch.Tensor | None = None,
):
    """
    Generates [B, L, 3] token ids autoregressively.
    If x0 is provided, it seeds x[:,0,:] with real first tokens (recommended).
    Otherwise defaults to BOS=(0,0,0).
    """
    B = num_samples
    L = seq_len

    # output tokens
    x = torch.zeros((B, L, 3), dtype=torch.long, device=device)

    # seed first token if provided
    if x0 is not None:
        if x0.ndim != 2 or x0.shape[0] != B or x0.shape[1] != 3:
            raise ValueError(f"x0 must be [B,3], got {tuple(x0.shape)}")
        x[:, 0, :] = x0.to(device=device, dtype=torch.long)

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

    # recommended: seed first token from real data to avoid BOS bias
    ap.add_argument("--seed_from_real", action="store_true",
                    help="Initialize x[:,0,:] from real first-token distribution instead of (0,0,0).")
    ap.add_argument("--root_shards", default="window_shards",
                    help="Root directory containing benchmark/{train,val,test} NPZ shards.")
    ap.add_argument("--benchmark", default="compress-gzip",
                    help="Benchmark name under root_shards.")
    ap.add_argument("--seed_split", default="train", choices=["train", "val", "test"],
                    help="Which split to sample the first token from.")
    ap.add_argument("--max_seed_shards", type=int, default=10,
                    help="How many shards to scan when building the seed distribution.")

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

    embed.eval()
    ar_model.eval()
    heads.eval()

    # build optional x0 seed from real shards
    x0 = None
    if args.seed_from_real:
        x0_np = sample_first_tokens_from_real(
            root_shards=args.root_shards,
            benchmark=args.benchmark,
            split=args.seed_split,
            B=args.num_samples,
            max_shards=args.max_seed_shards,
            seed=args.seed,
        )
        x0 = torch.from_numpy(x0_np)

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
        x0=x0,
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
