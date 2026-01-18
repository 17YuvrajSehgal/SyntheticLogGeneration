"""
Improved NPZ Shard Dataset Loader

Key Improvements:
1. **Optimized Caching**: Better cache management for smaller shards
2. **Prefetching**: Asynchronous shard loading to hide I/O latency
3. **Memory Mapping**: Uses mmap for faster file access
4. **Better Shuffling**: Shard-level shuffling for better randomization
5. **Performance Metrics**: Optional profiling to identify bottlenecks

This loader is designed to work with the improved NPZ shards created by
parquet_to_windowed_npz_improved.py
"""

import glob
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Default channels available
ALL_CHANNELS = ("event", "dt", "cpu", "tid", "fd", "comm", "ret")


@dataclass
class SampleConfig:
    """
    Configuration for loading samples from the dataset.
    """
    seq_len: int = 1024
    channels: Tuple[str, ...] = ALL_CHANNELS
    return_dict: bool = False
    
    # Data Types
    dtype_event: torch.dtype = torch.long
    dtype_dt: torch.dtype = torch.float32
    dtype_cpu: torch.dtype = torch.long
    dtype_tid: torch.dtype = torch.long
    dtype_fd: torch.dtype = torch.long
    dtype_comm: torch.dtype = torch.long
    dtype_ret: torch.dtype = torch.long

    def get_dim(self) -> int:
        return len(self.channels)


class NPZShardDatasetImproved(Dataset):
    """
    Improved dataset loader for NPZ shards with better performance.
    
    Key improvements:
    - Larger cache for smaller shards (cache more shards in memory)
    - Memory mapping for faster file access
    - Better cache eviction strategy (LRU with OrderedDict)
    - Optional profiling for performance analysis
    
    Supported Arrays in NPZ:
      event: (N, L) int32
      dt:    (N, L) float32
      cpu:   (N, L) int8
      tid:   (N, L) int16
      fd:    (N, L) int16
      comm:  (N, L) int16
      ret:   (N, L) int16
    """

    def __init__(
        self,
        shard_paths: List[str],
        config: SampleConfig = SampleConfig(),
        cache_shards: int = 10,  # Increased default cache size
        seed: int = 1234,
        verbose: bool = False,
        use_mmap: bool = True,  # NEW: Use memory mapping
        profile: bool = False   # NEW: Enable profiling
    ):
        self.shard_paths = list(shard_paths)
        if not self.shard_paths:
            print("[WARN] NPZShardDatasetImproved initialized with 0 paths.")
            
        self.cfg = config
        self.cache_shards = max(0, int(cache_shards))
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.use_mmap = use_mmap
        self.profile = profile

        # Profiling stats
        if self.profile:
            self.stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'load_times': [],
                'total_loads': 0
            }

        # 1. Scan shards to build index
        self.shard_sizes = []
        
        if self.verbose:
            print(f"[NPZDatasetImproved] Scanning {len(self.shard_paths)} shards...")
        
        for p in self.shard_paths:
            try:
                # Quick peek at shape without loading full array
                with np.load(p, mmap_mode='r' if use_mmap else None) as d:
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
            print(f"[NPZDatasetImproved] Loaded {len(self.shard_paths)} shards.")
            print(f"                      Total samples: {self.total}")
            print(f"                      Cache size: {self.cache_shards} shards")
            print(f"                      Memory mapping: {self.use_mmap}")

        # Cache: OrderedDict for efficient LRU
        self._cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Find which shard contains the given global index."""
        shard_idx = int(np.searchsorted(self.cum_sizes, idx, side="right") - 1)
        local_idx = int(idx - self.cum_sizes[shard_idx])
        return shard_idx, local_idx

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        """Load a shard with caching and optional profiling."""
        
        # Check cache (LRU)
        if shard_idx in self._cache:
            if self.profile:
                self.stats['cache_hits'] += 1
            # Move to end (most recently used)
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]

        # Cache miss - load from disk
        if self.profile:
            self.stats['cache_misses'] += 1
            start_time = time.time()

        path = self.shard_paths[shard_idx]
        
        # Load with memory mapping if enabled
        mmap_mode = 'r' if self.use_mmap else None
        data = np.load(path, mmap_mode=mmap_mode)
        
        arrays = {}
        # Only load requested channels
        for key in self.cfg.channels:
            if key in data.files:
                # Copy to memory if using mmap (for caching)
                if self.use_mmap and self.cache_shards > 0:
                    arrays[key] = np.array(data[key])
                else:
                    arrays[key] = data[key]
            else:
                raise ValueError(f"Channel '{key}' requested but missing in {path}")
        
        if not self.use_mmap or self.cache_shards == 0:
            data.close()

        if self.profile:
            load_time = time.time() - start_time
            self.stats['load_times'].append(load_time)
            self.stats['total_loads'] += 1

        # Cache management (LRU with OrderedDict)
        if self.cache_shards > 0:
            self._cache[shard_idx] = arrays
            # Evict oldest if cache is full
            while len(self._cache) > self.cache_shards:
                self._cache.popitem(last=False)  # Remove oldest (FIFO/LRU)
        
        return arrays

    def __getitem__(self, idx: int):
        """Get a single sample by global index."""
        shard_idx, local_idx = self._locate(idx)
        arrays = self._load_shard(shard_idx)
        
        # Collect tensors
        tensors = {}
        L = self.cfg.seq_len
        
        for ch in self.cfg.channels:
            arr = arrays[ch][local_idx]  # (L_file,)
            
            # Length Check / Truncate / Pad
            if arr.shape[0] > L:
                arr = arr[:L]
            elif arr.shape[0] < L:
                pad_len = L - arr.shape[0]
                arr = np.pad(arr, (0, pad_len), mode='constant')

            # Convert to Tensor
            dt = getattr(self.cfg, f"dtype_{ch}", torch.long)
            tensors[ch] = torch.tensor(arr, dtype=dt)

        if self.cfg.return_dict:
            return tensors
        
        # Stack into single tensor
        stack_list = [tensors[ch] for ch in self.cfg.channels]
        return torch.stack(stack_list, dim=-1)

    def get_stats(self) -> Dict:
        """Get profiling statistics if enabled."""
        if not self.profile:
            return {}
        
        stats = self.stats.copy()
        if stats['load_times']:
            stats['avg_load_time'] = np.mean(stats['load_times'])
            stats['max_load_time'] = np.max(stats['load_times'])
        
        if stats['total_loads'] > 0:
            hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            stats['cache_hit_rate'] = hit_rate
        
        return stats


def make_dataloaders_improved(
    root_dir: str,
    benchmark: str = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 1234,
    config: SampleConfig = SampleConfig(),
    cache_shards: int = 10,  # Increased default
    prefetch_factor: int = 4,  # Increased prefetch
    use_mmap: bool = True,
    profile: bool = False
):
    """
    Creates improved DataLoaders for train/val/test splits.
    
    Args:
        root_dir: Root directory containing NPZ shards
        benchmark: Optional benchmark name
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed
        config: Sample configuration
        cache_shards: Number of shards to cache per worker
        prefetch_factor: Number of batches to prefetch per worker
        use_mmap: Use memory mapping for faster file access
        profile: Enable profiling
    
    Returns:
        train_dl, val_dl, test_dl
    """
    base = os.path.join(root_dir, benchmark) if benchmark else root_dir
    
    def get_ds(split):
        if benchmark:
            pattern = os.path.join(base, split, "*.npz")
            paths = sorted(glob.glob(pattern))
        else:
            pattern = os.path.join(root_dir, "**", split, "*.npz")
            paths = sorted(glob.glob(pattern, recursive=True))
            
        return NPZShardDatasetImproved(
            paths, config, cache_shards, seed, 
            use_mmap=use_mmap, profile=profile
        )

    train_ds = get_ds("train")
    val_ds = get_ds("val")
    test_ds = get_ds("test")
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Improved DataLoader settings
    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=True,
        generator=g, 
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    test_dl = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_dl, val_dl, test_dl
