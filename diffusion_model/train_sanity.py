import os
import torch
from torch.optim import AdamW

from synthetic_data_processing.npz_shard_dataset import make_dataloaders
from synthetic_data_processing.trace_embedding import TraceEmbedding
from diffusion_model.ddpm_train import DDPM, train_one_step
from diffusion_model.diffusion_denoiser import TransformerDenoiser


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Paths
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    SHARDS = os.path.join(ROOT, "window_shards")

    # Data
    train_loader, _, _ = make_dataloaders(
        root_dir=SHARDS,
        benchmark="compress-gzip",
        batch_size=32,
        num_workers=0,         # set >0 on Linux later
        pin_memory=False,
        seq_len=200,
        cache_shards=1,
    )

    # Vocab sizes from your inspection
    num_events = 301
    num_dt_buckets = 256
    num_cpus = 4

    d_model = 128

    embed = TraceEmbedding(
        num_events=num_events,
        num_dt_buckets=num_dt_buckets,
        num_cpus=num_cpus,
        d_model=d_model,
        dropout=0.1,
    ).to(device)

    denoiser = TransformerDenoiser(
        d_model=d_model, nhead=8, num_layers=4, dropout=0.1
    ).to(device)

    # Diffusion
    diffusion = DDPM(T=1000, device=device)

    # Optimizer (both embed + denoiser for sanity)
    optimizer = AdamW(list(embed.parameters()) + list(denoiser.parameters()), lr=2e-4)

    # Train a few steps
    steps = 200
    it = iter(train_loader)
    for step in range(1, steps + 1):
        try:
            x_ids = next(it)
        except StopIteration:
            it = iter(train_loader)
            x_ids = next(it)

        x_ids = x_ids.to(device)
        x0 = embed(x_ids)  # [B, L, d_model]

        loss = train_one_step(denoiser, diffusion, x0, optimizer)

        if step % 20 == 0:
            print(f"step {step:04d} | loss {loss:.4f}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
