# Synthetic Kernel Trace Dataset (Windowed NPZ Format)

This repository contains a **model-ready version of kernel execution traces** built from the public dataset **“LTTng Execution traces for ten Phoronix benchmarks (part1)”**. In our experiments we currently use **one benchmark: `compress-gzip`**, and specifically the **`all-events` tracing configuration** (kernel events).

The goal of this dataset is to convert large, raw kernel traces into a compact format that sequence models can learn from and generate: **event sequences + timing gaps + CPU IDs**.

---

## 1) Original data source (what we start from)

The Zenodo dataset provides LTTng traces for multiple Phoronix benchmarks (compress-gzip, ffmpeg, iozone, phpbench, …). For each benchmark there are:

* **32 runs** (`run0` … `run31`)
* **3 tracing configurations**:

  * `all-events` = kernel events (this is what we use)
  * `libc` = malloc/free and other libc calls
  * `perf-trace` = performance counters

Each run is an LTTng trace directory (CTF format), not directly usable by common ML pipelines.

---

## 2) What our final dataset contains

The final dataset consists of many compressed `.npz` files (shards). Each `.npz` stores a large number of fixed-length “windows” (subsequences) taken from one run.

Each window is represented by **three aligned token sequences**:

* `event` : **event type ID** (integer token)
* `dt` : **time-gap bucket ID** (integer token representing the delta-time since the previous event)
* `cpu` : **CPU ID** (integer token)

Shape and types per `.npz` shard:

* `event`: `int32` array of shape **[N, L]**
* `dt`: `uint8` array of shape **[N, L]**
* `cpu`: `uint8` array of shape **[N, L]**

Where:

* **L** = window length (default **200 events**)
* **N** = number of windows in the shard (up to **100,000** by default)

This means a model does not see raw text or timestamps. It sees a token stream:

> “Event type #8 happened, after dt-bucket #59, on CPU #1 …”

---

## 3) How the dataset is created (pipeline overview)

### Step A — LTTng trace → text (babeltrace2 dump)

You first convert each run’s LTTng trace into a plain text file using `babeltrace2`.

* Input: a run’s LTTng kernel directory (CTF trace)
* Output: one `.txt` file per run, containing one event per line

Each line looks like:

`[timestamp] (+delta) host event_name: { cpu_id = X }, { ...payload... }`

This text is the “raw” format used by the Python preprocessing scripts below.

*(This is done by your Slurm dump script; it runs `babeltrace2` over every `run*/kernel` and writes `runX.txt`.)*

---

### Step B — Build an event vocabulary (`build_vocab.py`)

Kernel events are strings like `kmem_kmalloc`, `sched_switch`, etc. Models usually need integers, so this script builds a consistent mapping:

* Reads many `*-all-events-run*.txt` files under a root directory
* Extracts the **event name** (token right before `:`)
* Counts frequencies and assigns IDs

Outputs in `--out_dir`:

* `vocab.json` : `{event_name -> event_id}`
* `id_to_event.json` : `{event_id -> event_name}` (stored as string keys in JSON)
* `vocab_stats.json` : summary (counts, top events, etc.)

This is the global “dictionary” used to turn text event names into numeric tokens. 

---

### Step C — Text → Parquet (`txt_to_parquet.py`)

Text files are large and expensive to re-parse repeatedly. This script converts each run’s `.txt` into a structured Parquet table.

For each parsed event line it writes:

* `dataset` : benchmark name (e.g., `compress-gzip`)
* `run_id` : run label (e.g., `run0`)
* `line_idx` : line number in the text file
* `ts_str` : timestamp string from `[...]` (kept as text)
* `delta_s` : the `(+...)` value as **seconds** (float)
* `t_rel_ns` : cumulative relative time in **nanoseconds** (sum of deltas)
* `event_name` : event name string (e.g., `kmem_kmalloc`)
* `cpu_id` : CPU extracted from `{ cpu_id = X }` (or `-1` if missing)
* `raw_line` : original raw text line (for traceability/debug)

Important behavior:

* Lines that don’t match the expected numeric delta format are **skipped** (e.g., first lines that may show `(+?.?????????)`). 

---

### Step D — Parquet → sliding windows → NPZ shards (`make_window_shards.py`)

This script creates the **final training dataset**.

Input: Parquet files for each run.
Output: `.npz` shards containing many fixed-length windows.

#### 1) Windowing (how one training sample is made)

* Take the event stream in order
* Extract windows of length **L = 200**
* Move forward by **stride = 50**
* This creates overlapping sequences (better data efficiency)

Each emitted window becomes one training example.

#### 2) Time representation (dt bucketization)

This script does not store real-valued time. It **bucketizes** the per-event delta time into a categorical ID:

* Convert `dt_sec` → nanoseconds
* Apply `log1p` to compress the range
* Scale into `dt_buckets` (default **256**)
* Clip to `[0, 255]`

So the model learns time as a discrete token like “bucket 73”, not “2.78 microseconds”. 

#### 3) Efficient streaming over big Parquet files

Parquet is read in large batches (`iter_batches`). Since windows can span across batch boundaries, the script keeps a **carry buffer** of the last `(window-1)` rows so no windows are lost and none are duplicated. 

#### 4) Train/val/test split (by run number)

The script parses the run number from the filename `runXX.parquet` and assigns:

* `run_id <= 23` → `train`
* `24–27` → `val`
* `28–31` → `test` 

#### 5) Output naming

Files are written like:

`<out_dir>/<benchmark>/<split>/runXX_shardYYYY.npz` 

#### Note about column names

`make_window_shards.py` expects Parquet columns named `event_id`, `dt_sec`, `cpu` by default, but `txt_to_parquet.py` outputs `event_name`, `delta_s`, `cpu_id`. In practice, you must either:

* run an intermediate renaming/mapping step, **or**
* call `make_window_shards.py` with `--event_col`, `--dt_col`, `--cpu_col` set to match what’s in your Parquet.

---

## 4) What a single token means (how to read the final NPZ)

For a window position `t`:

* `event[t]` = **which kernel event type happened**
* `dt[t]` = **bucket ID** representing the **time gap since the previous event** in the serialized stream
* `cpu[t]` = **which CPU the event was recorded on**

So each sample is a compact, discrete representation of kernel activity that is suitable for embedding-based sequence models and generative models.

---

## Included preprocessing scripts

* **`build_vocab.py`**: scans dumped text traces and builds `event_name ↔ event_id` mappings. 
* **`txt_to_parquet.py`**: parses babeltrace2 text lines into a structured Parquet table per run (includes `delta_s` and `cpu_id`). 
* **`make_window_shards.py`**: converts Parquet runs into windowed `.npz` shards; bucketizes delta time and performs train/val/test split by run ID.