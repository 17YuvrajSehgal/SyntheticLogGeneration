# train_autoregressive.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from synthetic_data_processing.npz_shard_dataset import make_dataloaders
from dataset.trace_embedding import TraceEmbedding
from diffusion_model.heads import TokenHeads  # reuse your heads
from autoregressive_transformer.autoregressive_transformer import ARTransformer


def save_ckpt(path, embed, ar_model, heads, optim, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "args": vars(args),
        "embed": embed.state_dict(),
        "ar_model": ar_model.state_dict(),
        "heads": heads.state_dict(),
        "optim": optim.state_dict(),
    }, path)


@torch.no_grad()
def eval_one_epoch(
    ar_model,
    heads,
    loader,
    device,
    num_events,
    num_dt_buckets,
    num_cpus,
    max_batches=50,
    w_event: float = 1.0,
    w_dt: float = 1.0,
    w_cpu: float = 1.0,
):
    ar_model.eval()
    heads.eval()

    total_loss = 0.0
    total_tokens = 0

    for bi, x_ids in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x_ids = x_ids.to(device, non_blocking=True)  # [B, L, 3]

        # teacher forcing: input is first L-1, target is next L-1
        x_in = x_ids[:, :-1, :]         # [B, L-1, 3]
        y    = x_ids[:, 1:, :]          # [B, L-1, 3]
        y_event, y_dt, y_cpu = y[..., 0], y[..., 1], y[..., 2]

        h = ar_model(x_in)              # [B, L-1, D]
        logits = heads(h)               # dict of [B, L-1, V]

        ce_event = F.cross_entropy(logits["event"].reshape(-1, num_events),      y_event.reshape(-1))
        ce_dt    = F.cross_entropy(logits["dt"].reshape(-1, num_dt_buckets),     y_dt.reshape(-1))
        ce_cpu   = F.cross_entropy(logits["cpu"].reshape(-1, num_cpus),          y_cpu.reshape(-1))

        loss = (w_event * ce_event) + (w_dt * ce_dt) + (w_cpu * ce_cpu)

        B, Lm1 = y_event.shape
        total_loss += loss.item() * (B * Lm1)
        total_tokens += (B * Lm1)

    return total_loss / max(1, total_tokens)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--benchmark", default="compress-gzip")
    ap.add_argument("--root_shards", default="window_shards")

    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--out_dir", default="checkpoints_ar")

    # vocab sizes (must match your shards)
    ap.add_argument("--num_events", type=int, default=301)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)

    # loss weights (recommended to reduce early CPU collapse)
    ap.add_argument("--w_event", type=float, default=1.0)
    ap.add_argument("--w_dt", type=float, default=1.0)
    ap.add_argument("--w_cpu", type=float, default=2.0)

    # misc
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--clip", type=float, default=1.0)

    # optional: log per-channel accuracies
    ap.add_argument("--log_acc", action="store_true", help="Print event/dt/cpu accuracies during training prints.")

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

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
        dropout=args.dropout,
    ).to(device)

    ar_model = ARTransformer(
        embed_module=embed,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    heads = TokenHeads(
        d_model=args.d_model,
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
    ).to(device)

    optim = AdamW(
        list(embed.parameters()) + list(ar_model.parameters()) + list(heads.parameters()),
        lr=args.lr
    )

    it = iter(train_loader)
    best_val = float("inf")
    best_path = None

    for step in range(1, args.steps + 1):
        try:
            x_ids = next(it)
        except StopIteration:
            it = iter(train_loader)
            x_ids = next(it)

        x_ids = x_ids.to(device, non_blocking=True)  # [B, L, 3]

        x_in = x_ids[:, :-1, :]        # [B, L-1, 3]
        y    = x_ids[:, 1:, :]         # [B, L-1, 3]
        y_event, y_dt, y_cpu = y[..., 0], y[..., 1], y[..., 2]

        ar_model.train()
        heads.train()

        h = ar_model(x_in)             # [B, L-1, D]
        logits = heads(h)

        ce_event = F.cross_entropy(logits["event"].reshape(-1, args.num_events),      y_event.reshape(-1))
        ce_dt    = F.cross_entropy(logits["dt"].reshape(-1, args.num_dt_buckets),     y_dt.reshape(-1))
        ce_cpu   = F.cross_entropy(logits["cpu"].reshape(-1, args.num_cpus),          y_cpu.reshape(-1))

        loss = (args.w_event * ce_event) + (args.w_dt * ce_dt) + (args.w_cpu * ce_cpu)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(embed.parameters()) + list(ar_model.parameters()) + list(heads.parameters()),
            args.clip
        )
        optim.step()

        if step % 50 == 0:
            if args.log_acc:
                with torch.no_grad():
                    pred_ev = torch.argmax(logits["event"], dim=-1)
                    pred_dt = torch.argmax(logits["dt"], dim=-1)
                    pred_cp = torch.argmax(logits["cpu"], dim=-1)
                    acc_ev = (pred_ev == y_event).float().mean().item()
                    acc_dt = (pred_dt == y_dt).float().mean().item()
                    acc_cp = (pred_cp == y_cpu).float().mean().item()

                print(
                    f"step {step:06d} | loss {loss.item():.4f} | "
                    f"ce_event {ce_event.item():.4f} ce_dt {ce_dt.item():.4f} ce_cpu {ce_cpu.item():.4f} | "
                    f"acc_event {acc_ev:.3f} acc_dt {acc_dt:.3f} acc_cpu {acc_cp:.3f}"
                )
            else:
                print(
                    f"step {step:06d} | loss {loss.item():.4f} | "
                    f"ce_event {ce_event.item():.4f} ce_dt {ce_dt.item():.4f} ce_cpu {ce_cpu.item():.4f}"
                )

        if step % args.eval_every == 0:
            val_nll = eval_one_epoch(
                ar_model, heads, val_loader, device,
                args.num_events, args.num_dt_buckets, args.num_cpus,
                max_batches=50,
                w_event=args.w_event,
                w_dt=args.w_dt,
                w_cpu=args.w_cpu,
            )
            print(f"[VAL] step {step:06d} | avg_nll_per_token {val_nll:.6f}")

            if val_nll < best_val:
                best_val = val_nll
                best_path = os.path.join(args.out_dir, args.benchmark, f"best_step_{step:06d}.pt")
                save_ckpt(best_path, embed, ar_model, heads, optim, step, args)
                print(f"[BEST] {best_path} (val_nll={best_val:.6f})")

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, args.benchmark, f"step_{step:06d}.pt")
            save_ckpt(ckpt_path, embed, ar_model, heads, optim, step, args)
            print("[SAVE]", ckpt_path)

    final_path = os.path.join(args.out_dir, args.benchmark, f"final_step_{args.steps:06d}.pt")
    save_ckpt(final_path, embed, ar_model, heads, optim, args.steps, args)
    print("[SAVE]", final_path)

    if best_path is not None:
        print(f"[DONE] best checkpoint: {best_path} (val_nll={best_val:.6f})")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
