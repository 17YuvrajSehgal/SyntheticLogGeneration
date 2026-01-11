# Synthetic Kernel Log Dataset (Enriched Parquet)

This directory contains the tools and metadata for generating and using the high-fidelity Synthetic Kernel Log dataset. The dataset is derived from LTTng kernel traces and stored in **Apache Parquet** format, optimized for Machine Learning tasks.

## 1. Dataset Structure

The processed data is organized by benchmark (dataset) and run ID.
- **Source**: `scratch/txt_traces_all_benchmarks/<benchmark>/<run>.txt`
- **Output**: `scratch/enriched_parquet/<benchmark>/<run>.parquet`

Each Parquet file corresponds to one trace execution (one "run").

## 2. Hybrid Schema Design

The dataset uses a **Hybrid Schema** strategy to balance query performance with complete data fidelity. It distinguishes between **Core** columns (dense), **First-Class** columns (sparse but structured), and a **Catch-All** column for rare parameters.

### A. Core Columns (Always Present)
These fields are guaranteed to exist for every event row.

| Column | Type | Description |
| :--- | :--- | :--- |
| `line_idx` | `int64` | Original line number in the text trace (0-indexed). |
| `ts_str` | `string` | Raw timestamp string (HH:MM:SS.ns). |
| `delta_s` | `float64` | Time elapsed since the previous event (seconds). The first event is `0.0`. |
| `t_rel_ns` | `int64` | Cumulative relative time in nanoseconds (monotonic). |
| `event_name` | `string` | Name of the kernel event (e.g., `sched_switch`). |
| `event_id` | `int32` | Integer ID mapped via `metadata_all_events/vocab.json`. |
| `cpu_id` | `int32` | CPU core ID where the event occurred. |

### B. First-Class Payload Columns (Sparse)
These are high-frequency, high-importance parameters extracted into dedicated columns. If an event does not possess these fields, the value is `null`.

| Column | Type | Description |
| :--- | :--- | :--- |
| `tid` | `int64` | Thread ID (Target TID for sched events, or current TID). |
| `pid` | `int64` | Process ID. |
| `comm` | `string` | Process Command Name (e.g., `python`, `swapper/0`). |
| `ret` | `int64` | Return value (common in exit syscalls). |
| `fd` | `int64` | File Descriptor. |
| `filename` | `string` | Path or filename involved in the operation. |
| `flags` | `string` | Flags associated with the event (e.g., open flags). |
| `state` | `int64` | Process state (common in scheduler events). |

### C. Catch-All Parameter
| Column | Type | Description |
| :--- | :--- | :--- |
| `extra_params` | `string` | JSON-encoded dictionary containing **all other** parameters not listed above (e.g., `{"entropy_count": 42}`). |

## 3. Usage

### Loading Data (Python/Pandas)
```python
import pandas as pd

# Load a single run
df = pd.read_parquet("path/to/enriched_parquet/txt_traces_mysql/run0.parquet")

# Inspect structured columns
print(df[['ts_str', 'event_name', 'cpu_id', 'tid', 'comm']].head())

# Parse extra parameters
import json
df['extras'] = df['extra_params'].apply(lambda x: json.loads(x) if x else {})
```

### Metadata
- **Vocabulary**: `metadata_all_events/vocab.json` contains the mapping from `event_name` to `event_id`.

## 4. Generation Process

The dataset is generated using the `synthetic_data_processing/txt_to_enriched_parquet.py` script.

**Command:**
```bash
python synthetic_data_processing/txt_to_enriched_parquet.py \
    --input <input_text_trace.txt> \
    --output <output_file.parquet> \
    --vocab dataset/metadata_all_events/vocab.json
```

**Cluster Execution:**
Slurm jobs are provided in `slurm_jobs/enriched_parquet_jobs/` to parallelize processing across datasets.
