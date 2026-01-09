import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from synthetic_data_processing.npz_shard_dataset import make_dataloaders
from dataset.trace_embedding import TraceEmbedding

from diffusion_model.diffusion_denoiser import TransformerDenoiser
from diffusion_model.ddpm_train import DDPM, diffusion_step_losses
from diffusion_model.heads import TokenHeads


def save_ckpt(path, embed, denoiser, heads, optim, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "args": vars(args),
        "embed": embed.state_dict(),
        "denoiser": denoiser.state_dict(),
        "heads": heads.state_dict(),
        "optim": optim.state_dict(),
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="compress-gzip")
    ap.add_argument("--root_shards", default="window_shards")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--out_dir", default="checkpoints_discrete")

    # loss weights
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_event", type=float, default=1.0)
    ap.add_argument("--w_dt", type=float, default=1.0)
    ap.add_argument("--w_cpu", type=float, default=1.0)

    # vocab sizes (your values)
    ap.add_argument("--num_events", type=int, default=301)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Resolve paths from project root (works when running with -m)
    ROOT = os.getcwd()
    shards_root = os.path.join(ROOT, args.root_shards)

    train_loader, val_loader, _ = make_dataloaders(
        root_dir=shards_root,
        benchmark=args.benchmark,
        batch_size=args.batch_size,
        num_workers=args.num_workers if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
        seq_len=args.seq_len,
        cache_shards=1,
    )

    print("Train shards:", len(train_loader.dataset.shard_paths))
    print("Total windows:", len(train_loader.dataset))

    embed = TraceEmbedding(
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
        d_model=args.d_model,
        dropout=0.1,
    ).to(device)

    denoiser = TransformerDenoiser(
        d_model=args.d_model, nhead=8, num_layers=6, dropout=0.1
    ).to(device)

    heads = TokenHeads(
        d_model=args.d_model,
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
    ).to(device)

    diffusion = DDPM(T=args.T, device=device)

    optim = AdamW(
        list(embed.parameters()) + list(denoiser.parameters()) + list(heads.parameters()),
        lr=args.lr
    )

    it = iter(train_loader)

    for step in range(1, args.steps + 1):
        try:
            x_ids = next(it)
        except StopIteration:
            it = iter(train_loader)
            x_ids = next(it)

        x_ids = x_ids.to(device, non_blocking=True)  # [B, L, 3]
        # targets
        y_event = x_ids[..., 0]
        y_dt    = x_ids[..., 1]
        y_cpu   = x_ids[..., 2]

        x0 = embed(x_ids)  # [B, L, D]

        # diffusion timestep per sample
        t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device, dtype=torch.long)

        mse, x0_hat, _ = diffusion_step_losses(denoiser, diffusion, x0, t)

        logits = heads(x0_hat)
        ce_event = F.cross_entropy(logits["event"].reshape(-1, args.num_events), y_event.reshape(-1))
        ce_dt    = F.cross_entropy(logits["dt"].reshape(-1, args.num_dt_buckets), y_dt.reshape(-1))
        ce_cpu   = F.cross_entropy(logits["cpu"].reshape(-1, args.num_cpus), y_cpu.reshape(-1))

        loss = (
            args.w_mse   * mse +
            args.w_event * ce_event +
            args.w_dt    * ce_dt +
            args.w_cpu   * ce_cpu
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(embed.parameters()) + list(denoiser.parameters()) + list(heads.parameters()),
            1.0
        )
        optim.step()

        if step % 50 == 0:
            print(
                f"step {step:06d} | "
                f"loss {loss.item():.4f} | mse {mse.item():.4f} | "
                f"ce_event {ce_event.item():.4f} ce_dt {ce_dt.item():.4f} ce_cpu {ce_cpu.item():.4f}"
            )

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, args.benchmark, f"step_{step:06d}.pt")
            save_ckpt(ckpt_path, embed, denoiser, heads, optim, step, args)
            print("[SAVE]", ckpt_path)

    # final save
    ckpt_path = os.path.join(args.out_dir, args.benchmark, f"final_step_{args.steps:06d}.pt")
    save_ckpt(ckpt_path, embed, denoiser, heads, optim, args.steps, args)
    print("[SAVE]", ckpt_path)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
