***

# Dataset README — Windowed Kernel Trace Dataset (LTTng → Parquet → NPZ)

This repository provides a model-ready representation of Linux kernel execution traces derived from the public Zenodo dataset **“LTTng Execution traces for ten Phoronix benchmarks (part1)”**. The goal is to convert large raw traces into compact, fixed-length token sequences suitable for training sequence models (generation, anomaly detection, representation learning).

***

## 1) Original data source (Zenodo)

The source dataset contains LTTng execution traces collected for multiple Phoronix benchmarks (e.g., `compress-gzip`, `ffmpeg`, `iozone`, `phpbench`, etc.). For each benchmark, there are **32 runs** and **3 tracing configurations**:
- **all-events**: kernel events (this repo uses this)
- **libc**: memory-related libc calls (`malloc`, `free`, …)
- **perf-trace**: hardware performance counters

***

## 2) What the final dataset looks like

The final training data is stored as many compressed NumPy archives (`.npz`) called **shards**. Each shard contains many fixed-length windows extracted from one run.

Each `.npz` contains 3 arrays:
- `event`: `(N, L)` int32 — event type IDs
- `dt`: `(N, L)` uint8 — bucketized time gaps
- `cpu`: `(N, L)` uint8 — CPU IDs

Where:
- `L` = window length (default 200 events)
- `N` = number of windows stored in that shard (often 100,000)

This means the model does not see raw strings or full event payloads; it sees a compact token stream:
> “event type ID happened, after dt-bucket ID, on CPU ID”.

***

## 3) Understanding one raw `.txt` trace line

The raw trace is typically dumped into text (e.g., with `babeltrace2`) so each line is one event. Example:

```text
[08:46:31.491349294] (+0.000002786) flutin kmem_kfree: { cpu_id = 1 }, { call_site = 0x..., ptr = 0x... }
```

Key parts:
- `[08:46:31.491349294]` = absolute timestamp (human readable)
- `(+0.000002786)` = delta time since previous event (seconds)
- `flutin` = host name
- `kmem_kfree` = event name (kernel event type)
- `{ cpu_id = 1 }` = CPU ID (sometimes appears inside payload)
- Remaining key/value pairs are event-specific payload (pointers, sizes, syscall args, etc.)

***

## 4) Two Parquet “levels” (why you may see different columns)

You may encounter two kinds of Parquet files:

### A) “Rich parse” Parquet (debug/traceability oriented)
A general converter can store many fields such as:
- dataset name, run id, line index
- timestamp string
- delta seconds
- cumulative time in ns
- event name string
- cpu id
- raw original line

This is helpful for traceability, debugging parsing issues, and potential feature engineering.

### B) “Compact model” Parquet (what your notebook shows)
In this repo’s pipeline, the Parquet you inspect likely has only 4 columns:

- `t_sec` : absolute time in seconds (float)
- `dt_sec`: delta time in seconds (float)
- `cpu`   : CPU ID (int)
- `event_id`: integer ID for event type (int)

This compact Parquet is already “ML-oriented” (strings/payload removed, event names already mapped to IDs).

***

## 5) Vocabulary: mapping event names → event IDs

Kernel event names are strings (`sched_switch`, `kmem_kmalloc`, etc.). ML models work better with integer tokens, so a vocabulary is built:

- `vocab.json`: `eventname → event_id`
- `idtoevent.json`: `event_id → eventname`

This ensures the same event always maps to the same integer.

***

## 6) Windowing: what is a “window” and why it exists?

A single run is a very long event stream (millions of events). Training directly on one giant sequence is impractical.

A **window** is a fixed-length subsequence (default `L = 200`) of consecutive events:
- Window = 200 time steps
- Each time step includes `(event_id, dt, cpu)` tokens

Every window is treated as one training example.

### Why windowing is used
- Models typically require fixed-size batches
- Very long traces do not fit comfortably in memory/GPU training pipelines
- Windowing produces many independent samples for efficient training

***

## 7) Stride & overlap (small example)

**Stride** controls how far you move forward before taking the next window.

If:
- window length `L = 4`
- stride `S = 2`
- stream is: `E0 E1 E2 E3 E4 E5 E6 E7 E8 E9`

Then windows are:
- Window 0 (start 0): `[E0 E1 E2 E3]`
- Window 1 (start 2): `[E2 E3 E4 E5]`
- Window 2 (start 4): `[E4 E5 E6 E7]`
- Window 3 (start 6): `[E6 E7 E8 E9]`

Overlap happens because `S < L`.  
Overlap length = `L - S` = `4 - 2 = 2` events shared between neighboring windows.

In this repo the defaults are typically:
- `L = 200`
- `S = 50`
So windows overlap heavily, increasing the number of training examples.

***

## 8) Shards: what is a “shard” and why it exists?

A **shard** is a single `.npz` file that contains many windows (not just one). This avoids having millions of tiny files and makes training I/O faster.

Example shard contents (your inspection output):

```text
event: (100000, 200) int32
dt   : (100000, 200) uint8
cpu  : (100000, 200) uint8
```

Meaning:
- The shard contains **100,000 windows**
- Each window is **200 events long**
- Each window position has aligned `event`, `dt`, and `cpu` tokens

A run can generate multiple shards:
- `run00_shard0000.npz`
- `run00_shard0001.npz`
- …
because the generator writes a shard whenever it accumulates `shardsize` windows (often 100,000).

***

## 9) Time representation: `dt_sec` → `dt` buckets

Instead of storing raw float `dt_sec` in the final `.npz`, the pipeline converts time gaps into discrete buckets:
- Convert seconds → nanoseconds
- Apply `log1p` compression (to handle long-tailed timing distributions)
- Scale into a fixed number of buckets (often 256)
- Clip into `[0, 255]`
- Store as `uint8`

So `dt` is a categorical token like “bucket 73”, not “2.78 microseconds”.

***

## 10) Output directory structure (example)

Typical structure:

```text
window_shards/
  compress-gzip/
    train/
      run00_shard0000.npz
      run00_shard0001.npz
      ...
    val/
      run24_shard0000.npz
      ...
    test/
      run28_shard0000.npz
      run28_shard0001.npz
      ...
      run31_shard0004.npz
```

Meaning:
- `compress-gzip` is the benchmark name
- `train/val/test` are dataset splits
- `runXX` indicates which original run produced these windows
- `shardYYYY` is the shard index for that run

### Train/val/test split strategy
Splits are done by **run number**, not by randomly mixing windows. This prevents leakage where nearly-identical overlapping windows from the same run appear in both train and test.

***

## 11) Example: inspecting a shard

Here is a minimal script to inspect `.npz` shards:

```python
import numpy as np

def inspect_npz(npz_path, n=3, k=15):
    d = np.load(npz_path)
    event = d["event"]
    dt    = d["dt"]
    cpu   = d["cpu"]

    print("=== SHAPES ===")
    print("event:", event.shape, event.dtype)
    print("dt   :", dt.shape, dt.dtype)
    print("cpu  :", cpu.shape, cpu.dtype)

    B, L = event.shape

    print("\n=== SAMPLE SEQUENCES (first 20 tokens) ===")
    for i in range(min(n, B)):
        print(f"\n-- seq {i} --")
        print("event:", event[i, :20].tolist())
        print("dt   :", dt[i, :20].tolist())
        print("cpu  :", cpu[i, :20].tolist())

    print("\n=== BASIC STATS ===")
    print("event min/max:", int(event.min()), int(event.max()))
    print("dt    min/max:", int(dt.min()), int(dt.max()))
    print("cpu   min/max:", int(cpu.min()), int(cpu.max()))

    ev_counts = np.bincount(event.reshape(-1))
    top = np.argsort(ev_counts)[::-1][:k]

    print(f"\nTop {k} events:")
    for eid in top:
        if ev_counts[eid] == 0:
            break
        print(f"  event_id={eid}: {int(ev_counts[eid])}")

    cpu_counts = np.bincount(cpu.reshape(-1))
    print("\nCPU distribution:")
    for c in range(len(cpu_counts)):
        if cpu_counts[c] > 0:
            print(f"  cpu {c}: {int(cpu_counts[c])}")

    dt_flat = dt.reshape(-1).astype(np.int64)
    print("\nDT bucket stats:")
    print(
        "  min=", int(dt_flat.min()),
        "max=", int(dt_flat.max()),
        "mean=", float(dt_flat.mean()),
    )

npz_path = r"window_shards\compress-gzip\train\run00_shard0000.npz"
inspect_npz(npz_path, n=5, k=20)
```

***

## 12) Should we store more columns than (event, dt, cpu)?
The current dataset design intentionally stores only the minimal information needed for general-purpose sequence modeling:
- event type (what)
- time gap (when)
- CPU (where)

From raw traces, you *could* add extra features (e.g., `tid`, `comm`, `bytes_alloc`, `target_cpu`) but that increases complexity and often requires tokenization/bucketing strategies similar to `dt`. A good practice is:
- Start with the minimal dataset (event/dt/cpu)
- Add one feature family at a time only if the downstream task requires it

***

## 13) Notes / gotchas
- The very first line in a run can have an unknown delta `(+?.?????????)`. Pipelines typically skip or handle these specially.
- Raw payload values like pointers (`ptr`, `call_site`) are extremely high-cardinality and often not useful as direct tokens without heavy normalization.

***