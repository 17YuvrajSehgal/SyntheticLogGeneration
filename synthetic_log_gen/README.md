# synthetic_log_gen

This package contains the core components for the Synthetic Kernel Log Generation framework. It is divided into **Data Loading** and **Model Architecture**.

## 1. Data Loading (`synthetic_log_gen.data`)

### [dataset.py](data/dataset.py)
This module handles the efficient loading of large-scale kernel trace datasets stored in `.npz` format.

*   **`NPZShardDataset`**: A PyTorch `IterableDataset` designed for low-memory streaming.
    *   **Caching**: Loads `.npz` files (shards) on demand and caches a metadata index to avoid re-scanning disk.
    *   **Features**: Returns a dictionary of tensors (`event`, `dt`, `cpu`, `tid`, etc.) for each window.
    *   **Config**: Uses `SampleConfig` to control sequence length (`seq_len`) and stride.

*   **`make_dataloaders`**:
    *   Automatically discovers dataset partitions (`train`/`val`/`test`).
    *   Supports **Recursive Discovery**: If no specific benchmark is requested, it recursively loads all `.npz` files from all subdirectories, enabling "All-Benchmark" training.

---

## 2. Model Architecture (`synthetic_log_gen.models`)

The model is a **Transformer-based Diffusion Model (DDPM)** designed for multi-modal categorical data (Events, TIDs, etc.) and continuous data (Time Deltas).

### [diffusion.py](models/diffusion.py)
The main entry point for the model.

*   **`LogDiffusionModel`**:
    *   **End-to-End Flow**: Embeds input $\rightarrow$ Adds Noise (Forward Process) $\rightarrow$ Denoises (Reverse Process) $\rightarrow$ Projects back to Vocabulary.
    *   **Loss Function**: Hybrid loss combining:
        1.  **Latent Loss (MSE)**: Difference between added noise and predicted noise.
        2.  **Reconstruction Loss (Cross Entropy/MSE)**: Auxiliary loss forcing the latent representation to retain semantic meaning of attributes (TID, CPU, etc.).
    *   **`sample()`**: Implements the reverse diffusion loop (1000 steps) to generate new traces from pure Gaussian noise.

*   **`TransformerDenoiser`**:
    *   The core backbone. A standard Transformer Encoder that takes noisy latent vectors $x_t$ and timestep embeddings $t$ to predict noise $\epsilon$.

### [embeddings.py](models/embeddings.py)
Handles the translation between raw log features and the continuous latent space of the Diffusion Model.

*   **`FeatureEmbedder`**:
    *   **Inputs**: Dictionary of raw indices (e.g., Event ID `42`, CPU `0`).
    *   **Operation**: Looks up learnable embeddings for each categorical feature (Event, TID, CPU, Comm, Ret) and projects `dt` (time delta) via a linear layer.
    *   **Fusion**: Concatenates all feature embeddings and projects them to `d_model` size.

*   **`FeatureUnembedder`**:
    *   **Inputs**: Denoised latent vectors $x_0$.
    *   **Operation**: Projects the latent vector back to the logits for each vocabulary (Event logits, CPU logits, etc.).
    *   **Purpose**: Allows calculating the "Reconstruction Loss" and decoding generated samples into readable traces.
