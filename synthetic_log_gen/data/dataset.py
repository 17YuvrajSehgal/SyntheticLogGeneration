import glob
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Default channels available in usage
ALL_CHANNELS = ("event", "dt", "cpu", "tid", "fd", "comm", "ret")

@dataclass
class SampleConfig:
    """
    Configuration for loading samples from the dataset.
    """
    seq_len: int = 1024
    # Which channels to load and return. Order matters for the stacked tensor.
    channels: Tuple[str, ...] = ALL_CHANNELS 
    return_dict: bool = False  # If True, returns dict of tensors. If False, returns stacked tensor.
    
    # Data Types
    dtype_event: torch.dtype = torch.long
    dtype_dt: torch.dtype = torch.float32  # Changed to float32 for log-time
    dtype_cpu: torch.dtype = torch.long
    dtype_tid: torch.dtype = torch.long
    dtype_fd: torch.dtype = torch.long
    dtype_comm: torch.dtype = torch.long
    dtype_ret: torch.dtype = torch.long

    def get_dim(self) -> int:
        return len(self.channels)


class NPZShardDataset(Dataset):
    """
    Dataset over many .npz shards. Each shard is expected to contain arrays aligned by index.
    
    Supported Arrays in NPZ:
      event: (N, L) int32
      dt:    (N, L) float32
      cpu:   (N, L) int8
      tid:   (N, L) int16
      fd:    (N, L) int16
      comm:  (N, L) int16
      ret:   (N, L) int16
      
    This dataset:
    - builds a global index: global_sample_idx -> (shard_idx, local_row_idx)
    - lazily loads shards
    - caches a few shards in memory (LRU)
    """

    def __init__(
        self,
        shard_paths: List[str],
        config: SampleConfig = SampleConfig(),
        cache_shards: int = 2,
        seed: int = 1234,
        verbose: bool = False
    ):
        self.shard_paths = list(shard_paths)
        if not self.shard_paths:
            # We allow empty for some edge cases but warn
            print("[WARN] NPZShardDataset initialized with 0 paths.")
            
        self.cfg = config
        self.cache_shards = max(0, int(cache_shards))
        self.rng = random.Random(seed)
        self.verbose = verbose

        # 1. Scan shards to build index
        self.shard_sizes = []
        
        # We only check 'event' key to determine size
        for p in self.shard_paths:
            try:
                # We use mmap_mode='r' to peek shapes without full load? 
                # Actually np.load with allow_pickle=False reads header.
                with np.load(p) as d:
                    if "event" not in d.files:
                        print(f"[WARN] Skipping {p}: missing 'event' array.")
                        self.shard_sizes.append(0)
                        continue
                    self.shard_sizes.append(int(d["event"].shape[0]))
            except Exception as e:
                print(f"[ERR] Failed to read {p}: {e}")
                self.shard_sizes.append(0)

        # Prefix sum for global indexing
        self.cum_sizes = np.cumsum([0] + self.shard_sizes)
        self.total = int(self.cum_sizes[-1])
        
        if self.verbose:
            print(f"[NPZDataset] Loaded {len(self.shard_paths)} shards. Total samples: {self.total}")

        # Cache: shard_idx -> dict of numpy arrays
        self._cache: Dict[int, Dict[str, np.ndarray]] = {} 
        self._cache_order: List[int] = []  # LRU

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        # Find shard index
        shard_idx = int(np.searchsorted(self.cum_sizes, idx, side="right") - 1)
        local_idx = int(idx - self.cum_sizes[shard_idx])
        return shard_idx, local_idx

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        # Check cache
        if shard_idx in self._cache:
            # Update LRU (move to end)
            if shard_idx in self._cache_order:
                self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        path = self.shard_paths[shard_idx]
        data = np.load(path)
        
        arrays = {}
        # Only load requested channels
        for key in self.cfg.channels:
            if key in data.files:
                arrays[key] = data[key]
            else:
                # Fallback / Warning? 
                # For training robustness, maybe raise error or fill zeros.
                # raising error is safer to detect mismatches.
                raise ValueError(f"Channel '{key}' requested but missing in {path}")
        
        data.close()

        # Cache management
        if self.cache_shards > 0:
            self._cache[shard_idx] = arrays
            self._cache_order.append(shard_idx)
            while len(self._cache_order) > self.cache_shards:
                evict = self._cache_order.pop(0)
                if evict in self._cache:
                    del self._cache[evict]
        
        return arrays

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(idx)
        arrays = self._load_shard(shard_idx)
        
        # Collect tensors
        tensors = {}
        L = self.cfg.seq_len
        
        for ch in self.cfg.channels:
            arr = arrays[ch][local_idx] # (L_file,)
            
            # Length Check / Truncate
            if arr.shape[0] > L:
                arr = arr[:L]
            elif arr.shape[0] < L:
                # Pad? Or Error? 
                # Usually we expect fixed length shards. 
                # Let's pad with 0s if needed, or raise.
                # Standard impl: Pad
                pad_len = L - arr.shape[0]
                arr = np.pad(arr, (0, pad_len), mode='constant')

            # Convert to Tensor
            # Map channel to dtype
            dt = getattr(self.cfg, f"dtype_{ch}", torch.long)
            tensors[ch] = torch.tensor(arr, dtype=dt)

        if self.cfg.return_dict:
            return tensors
        
        # Stack
        # Ensure identical shapes? (L,)
        # stack -> (L, C)
        stack_list = [tensors[ch] for ch in self.cfg.channels]
        return torch.stack(stack_list, dim=-1)


def make_dataloaders(
    root_dir: str,
    benchmark: str = None, # Optional, if None assumes root_dir contains train/val/test directly
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 1234,
    config: SampleConfig = SampleConfig(),
    cache_shards: int = 2
):
    """
    Creates DataLoaders for train/val/test splits.
    
    Structure expectation:
       root_dir/[train|val|test]/*.npz  (if benchmark is None)
       OR
       root_dir/benchmark/[train|val|test]/*.npz (if benchmark is set)
    """
    base = os.path.join(root_dir, benchmark) if benchmark else root_dir
    
    def get_ds(split):
        pattern = os.path.join(base, split, "*.npz")
        paths = sorted(glob.glob(pattern))
        # If no files, return empty DS? simpler to just pass empty list
        return NPZShardDataset(paths, config, cache_shards, seed)

    train_ds = get_ds("train")
    val_ds = get_ds("val")
    test_ds = get_ds("test")
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        generator=g, persistent_workers=(num_workers > 0)
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    
    return train_dl, val_dl, test_dl
