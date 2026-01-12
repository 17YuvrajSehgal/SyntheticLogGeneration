# Windowed NPZ Dataset Format

This directory contains the training data for the Diffusion Model, stored in compressed NumPy (`.npz`) format. This format is optimized for high-performance loading during training.

## 1. Overview
The dataset is derived from the **Enriched Parquet** files (see `README_parquet.md`) and pre-processed into fixed-length windows.

*   **Format**: Compressed NumPy Archive (`.npz`).
*   **Organization**: `dataset/windowed_npz_<SEQ_LEN>/<split>/*.npz`
*   **Content**: Each file contains multiple "windows" (sequences) of kernel trace events.

## 2. File Structure
Each `.npz` file contains a set of arrays. All arrays share the same first two dimensions: `(NumWindows, SeqLen)`.

| Key | Type | Shape | Description | Processing logic |
| :--- | :--- | :--- | :--- | :--- |
| **`event`** | `int32` | `(N, L)` | Event ID | Mapped from `metadata_all_events/vocab.json`. |
| **`dt`** | `float32` | `(N, L)` | Time Delta | Log-normalized: $\log(1 + \Delta t)$. Represents time since the previous event. |
| **`cpu`** | `int8` | `(N, L)` | CPU Core ID | Raw CPU ID (e.g., 0, 1, 2...). |
| **`tid`** | `int16` | `(N, L)` | Thread ID | Hashed: `tid % 256` (matches `tid_buckets` config). Preserves thread identity locally. |
| **`fd`** | `int16` | `(N, L)` | File Descriptor | Clamped: Values $\ge 1024$ are capped at 1024. |
| **`comm`** | `int16` | `(N, L)` | Command Name | Mapped from `metadata_all_events/vocab_comm.json`. Represents process name (e.g., `python`). |
| **`ret`** | `int16` | `(N, L)` | Return Value | Mapped from `metadata_all_events/vocab_ret.json`. Represents syscall result. |

*   **N**: Number of windows in this file.
*   **L**: Sequence Length (e.g., 256, 1024, 4096).

## 3. Data Transformation Pipeline

The data flows through the following pipeline before reaching this format:

1.  **Raw Text Trace** (`.txt`) $\rightarrow$ **Enriched Parquet** (`.parquet`)
    *   *Script*: `synthetic_data_processing/txt_to_enriched_parquet.py`
    *   *Action*: Parses text, extracts structured columns (`tid`, `comm`), and handles rare parameters.
    
2.  **Enriched Parquet** $\to$ **Windowed NPZ** (`.npz`)
    *   *Script*: `synthetic_data_processing/parquet_to_windowed_npz.py`
    *   *Action*:
        *   **Tokenization**: Maps strings (`comm`) to integers IDs.
        *   **Normalization**: Applies $\log(1+x)$ to time deltas.
        *   **Windowing**: Slides a window of length `L` with stride `S` over the trace.
        *   **Saving**: Aggregates windows and saves compressed.

## 4. Usage in Training

The `synthetic_log_gen.data.NPZShardDataset` class is responsible for loading these files.

```python
# Example Loading
import numpy as np
data = np.load("dataset/windowed_npz_1024/train/run1_L1024.npz")

events = data['event'] # Shape (1000, 1024)
times = data['dt']     # Shape (1000, 1024)
```

## 5. Notes on Values
*   **Padding**: `0` is used as the padding value for most integer fields.
*   **Unknowns**: Vocabulary lookups map unknown tokens to a specific `<UNK>` ID (usually `1`).
