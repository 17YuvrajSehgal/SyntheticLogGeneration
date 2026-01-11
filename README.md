# Synthetic Log Generation Framework

This repository contains the implementation of a **Diffusion Model for Kernel Trace Generation**. This document explains the core scripts for running experiments: **Training** (`train_experiment.py`) and **Generation** (`sample_diffusion.py`).

## 1. Training (`train_experiment.py`)

This is the main entry point for training the Diffusion Model. It handles the entire training lifecycle, from data loading to H100-optimized execution.

### **Workflow**
1.  **Data Loading**: Uses `LogDiffusionModel` and `make_dataloaders` to stream data from `.npz` shards. It automatically detects if mixed precision (`bf16`) or TF32 should be used.
2.  **Model Initialization**: Builds the `LogDiffusionModel` with a Transformer backbone (`d_model`, `nhead`, `layers`).
3.  **Optimization Loop**:
    *   Samples random timesteps $t$.
    *   Adds noise to the embeddings ($q(x_t|x_0)$).
    *   Predicts the noise using the Denoiser ($p_\theta(x_{t-1}|x_t)$).
    *   Optimizes a hybrid loss: `Latent Loss (MSE) + 0.1 * Reconstruction Loss`.
4.  **Logging**: Writes training metrics (Loss, Latent Loss, Recon Loss) to **TensorBoard** (`logs_tensorboard/`) and saves checkpoints per epoch.

### **Key Arguments**
*   `--data-root`: Path to the `.npz` dataset shards.
*   `--seq-len`: Window size (e.g., `256`, `1024`, `4096`).
*   `--mixed-precision bf16`: Essential for H100 performance.
*   `--compile`: Enables `torch.compile` (Graph Mode) for speed.

**Example Usage**:
```bash
python train_experiment.py --data-root data/windowed_256 --benchmark scimark2 --seq-len 256 --mixed-precision bf16
```

---

## 2. Sampling (`sample_diffusion.py`)

This script performs inference (generation) using a trained model checksum. It implements the **Reverse Diffusion Process**.

### **Workflow**
1.  **Model Setup**: Re-initializes the `LogDiffusionModel` with the *exact same configuration* used during training (`d_model`, `layers`, etc.).
2.  **Checkpoint Loading**: Loads the weights from a specific `.pt` file (e.g., `ckpt_epoch_99.pt`).
3.  **Generation Loop**:
    *   Starts with pure Gaussian Noise ($x_T \sim \mathcal{N}(0, I)$).
    *   Iteratively denoises for 1000 steps ($x_{1000} \to x_{999} \dots \to x_0$).
    *   Uses the learnable `Head` to project the final latent $x_0$ back to discrete tokens (Events, PIDs, etc.).
4.  **Output**: Saves the generated traces as a compressed `.npz` file.

### **Key Arguments**
*   `--ckpt`: Path to the trained model checkpoint.
*   `--out`: Destination `.npz` file for generated traces.
*   `--num-samples`: Number of traces to generate.
*   `--num-layers`: **MUST match training**. (e.g., `8` for H100 runs).

**Example Usage**:
```bash
python sample_diffusion.py --ckpt logs/experiment_1/ckpt_99.pt --out generated/traces.npz --num-layers 8 --steps 1000
```

---

## 3. Overall Experiment Flow

1.  **Prepare Data**: Convert Parquet logs to `.npz` windows (`parquet_to_windowed_npz.py`).
2.  **Train**: Run `train_experiment.py` on the GPU cluster (via Slurm).
3.  **Generate**: Run `sample_diffusion.py` to create synthetic traces from the trained model.
4.  **Evaluate**: Compare the statistics of the Generated `.npz` against the Real `.npz`.
