import glob
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SampleConfig:
    seq_len: int = 200
    channels: Tuple[str, ...] = ("event", "dt", "cpu")  # order of channels returned
    return_dict: bool = False  # if True -> {"event":..., "dt":..., "cpu":...}
    dtype_event: torch.dtype = torch.long
    dtype_dt: torch.dtype = torch.long     # treat dt buckets as categorical ids
    dtype_cpu: torch.dtype = torch.long    # treat cpu ids as categorical ids


class NPZShardDataset(Dataset):
    """
    Dataset over many .npz shards. Each shard stores arrays:
      event: (N, L) int32
      dt:    (N, L) uint8
      cpu:   (N, L) uint8

    This dataset:
    - builds a global index: global_sample_idx -> (shard_idx, local_row_idx)
    - caches one shard in memory for speed (LRU of size 1 by default)
    """

    def __init__(
        self,
        shard_paths: List[str],
        config: SampleConfig = SampleConfig(),
        cache_shards: int = 1,
        seed: int = 1234,
    ):
        self.shard_paths = list(shard_paths)
        if not self.shard_paths:
            raise ValueError("No shard paths provided.")

        self.cfg = config
        self.cache_shards = max(0, int(cache_shards))
        self.rng = random.Random(seed)

        # Read only the shapes to build indexing (cheap)
        self.shard_sizes = []
        for p in self.shard_paths:
            with np.load(p) as d:
                if "event" not in d.files:
                    raise ValueError(f"{p} missing 'event' array. Found {d.files}")
                self.shard_sizes.append(int(d["event"].shape[0]))

        # Prefix sum for global indexing
        self.cum_sizes = np.cumsum([0] + self.shard_sizes)  # len = num_shards+1
        self.total = int(self.cum_sizes[-1])

        # Simple cache: keep last K shards loaded
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}  # shard_idx -> arrays dict
        self._cache_order: List[int] = []  # LRU order

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        # Find shard index via binary search on cum_sizes
        shard_idx = int(np.searchsorted(self.cum_sizes, idx, side="right") - 1)
        local_idx = int(idx - self.cum_sizes[shard_idx])
        return shard_idx, local_idx

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        # Return from cache if present
        if shard_idx in self._cache:
            # update LRU
            if shard_idx in self._cache_order:
                self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        path = self.shard_paths[shard_idx]
        data = np.load(path)

        arrays = {}
        for key in ("event", "dt", "cpu"):
            if key in data.files:
                arrays[key] = data[key]
        data.close()

        # Add to cache
        if self.cache_shards > 0:
            self._cache[shard_idx] = arrays
            self._cache_order.append(shard_idx)
            # evict old
            while len(self._cache_order) > self.cache_shards:
                old = self._cache_order.pop(0)
                if old in self._cache:
                    del self._cache[old]

        return arrays

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(idx)
        arrays = self._load_shard(shard_idx)

        # Fetch row
        e = arrays["event"][local_idx]
        dt = arrays["dt"][local_idx]
        cpu = arrays["cpu"][local_idx]

        # Optionally enforce seq_len (should already match)
        L = self.cfg.seq_len
        if e.shape[0] != L:
            e = e[:L]
            dt = dt[:L]
            cpu = cpu[:L]

        # Convert to torch
        te = torch.tensor(e, dtype=self.cfg.dtype_event)
        tdt = torch.tensor(dt, dtype=self.cfg.dtype_dt)
        tcpu = torch.tensor(cpu, dtype=self.cfg.dtype_cpu)

        if self.cfg.return_dict:
            return {"event": te, "dt": tdt, "cpu": tcpu}

        # Stack channels -> [L, C] for diffusion
        chans = []
        for name in self.cfg.channels:
            if name == "event":
                chans.append(te)
            elif name == "dt":
                chans.append(tdt)
            elif name == "cpu":
                chans.append(tcpu)
            else:
                raise ValueError(f"Unknown channel '{name}'.")
        x = torch.stack(chans, dim=-1)  # [L, C]
        return x


def make_dataloaders(
    root_dir: str,
    benchmark: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 1234,
    seq_len: int = 200,
    cache_shards: int = 1,
    return_dict: bool = False,
):
    """
    root_dir/
      benchmark/
        train/*.npz
        val/*.npz
        test/*.npz
    """
    def list_shards(split: str) -> List[str]:
        pattern = os.path.join(root_dir, benchmark, split, "*.npz")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No shards found: {pattern}")
        return paths

    cfg = SampleConfig(seq_len=seq_len, return_dict=return_dict)

    train_ds = NPZShardDataset(list_shards("train"), config=cfg, cache_shards=cache_shards, seed=seed)
    val_ds   = NPZShardDataset(list_shards("val"),   config=cfg, cache_shards=cache_shards, seed=seed)
    test_ds  = NPZShardDataset(list_shards("test"),  config=cfg, cache_shards=cache_shards, seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=g,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
