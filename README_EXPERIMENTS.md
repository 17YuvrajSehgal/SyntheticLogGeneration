# Experiment Documentation: Diffusion Models for Kernel Trace Generation

This document details the experimental setup, configurations, and rationale for the research paper on **"Synthetic Kernel Log Generation using Diffusion Models"**. It is designed to allow reproducible research and continuation of the experiments.

## 1. Research Objectives

The primary goal is to demonstrate that **Diffusion Models** are superior to traditional approaches (RNNs/GANs) for generating system kernel traces, specifically in:
1.  **Semantic Fidelity**: Capturing complex cause-and-effect relationships (e.g., `PID` matching `SchedSwitch`).
2.  **Long-Range Dependency**: Generating valid causal chains that span thousands of events (e.g., `fopen` ... `fclose`).

We conduct two main ablation studies to prove these points.

---

## 2. Experiment 1: Feature Ablation Study

**Goal**: Demonstrate the necessity of "Rich System Metadata". We hypothesize that predicting only `Event` + `Time` is insufficient for valid trace generation, and that adding `CPU`, `TID`, and `Comm` significantly improves validity.

### Configurations

All configurations use **Sequence Length = 1024** and **Benchmark = scimark2**.

| Experiment Name | Channels Used | Rationale |
| :--- | :--- | :--- |
| **`ablation_base`** | `event`, `dt` | **Baseline**. Matches standard literature treatment of logs as simple text sequences. Expected to fail at thread logic. |
| **`ablation_system`** | `event`, `dt`, `cpu`, `tid` | **System-Aware**. Adds core scheduling context. Should capture parallelism but maybe not semantic intent. |
| **`ablation_full`** | `event`, `dt`, `cpu`, `tid`, `comm`, `ret` | **Feature-Rich**. Adds "Command Name" (Semantic Intent) and "Return Value" (Outcome). Expected to be state-of-the-art. |

---

## 3. Experiment 2: Context Length Study

**Goal**: Demonstrate the model's ability to scale to long contexts, which is critical for capturing macro-system behavior (e.g., garbage collection cycles, IO bursts) that short-context models miss.

### Configurations

All configurations use **Full Feature Set**.

| Experiment Name | Sequence Length | Batch Size | Hardware Notes                                                                                                             |
| :--- | :--- | :--- |:---------------------------------------------------------------------------------------------------------------------------|
| **`context_256`** | 256 | 256 | **Short Context**. Fast to train, but lacks global view.`256 seq` * `256 batch` = `65k tokens`                             |
| **`context_1024`** | 1024 | 128 | **Medium Context**. Standard balance for Transformer models.`1024 seq` * `128 batch` = `131k tokens`                       |
| **`context_4096`** | 4096 | 32 | **Long Context**. Requires H100 GPU (80GB). Validates scalability of the approach. `4096 seq` * `32 batch` = `131k tokens` |

**Rationale for Batch Sizes**:
Batch sizes were tuned to maximize H100 (80GB) memory utilization without OOM:
*   `256` fits easily with Batch=256.
*   `1024` (4x larger) requires reducing Batch to 128.
*   `4096` (16x larger) requires reducing Batch to 32.

---

## 4. Model Architecture & Hyperparameters

We use a **Transformer-based Denoising Diffusion Probabilistic Model (DDPM)**.

*   **Model Type**: `LogDiffusionModel` (Custom implementation in `synthetic_log_gen.models`).
*   **Dimensions**: `d_model=512`, `nhead=8`, `num_layers=8`.
*   **Optimizations (H100)**:
    *   **Mixed Precision**: `bf16` (BFloat16) for stable training at scale.
    *   **TF32**: Enabled for faster matrix multiplications.
    *   **Learning Rate**: `2e-4` with cosine decay.
    *   **Steps**: 1000 diffusion steps.

---

## 5. Directory Structure & Reproduction

### Directory Layout
*   `slurm_jobs/generate_experiments.py`: Script that generates the Slurm files.
*   `slurm_jobs/experiments/`: Contains the generated `.slurm` files for each experiment.
*   `experiments_results/`: **Output Directory**. Each experiment gets its own subfolder (e.g., `experiments_results/exp_context_4096/`) containing:
    *   `slurm_*.out/err`: Execution logs.
    *   `ckpt_epoch_*.pt`: Model checkpoints.
    *   `events.out.tfevents.*`: TensorBoard logs.

### How to Run
1.  **Generate Job Files**:
    ```bash
    python slurm_jobs/generate_experiments.py
    ```
2.  **Submit All Experiments**:
    ```bash
    bash run_experiments.sh
    ```
3.  **Generate Samples (Validation)**:
    Use `sample_diffusion.py` with the trained checkpoint.
    ```bash
    python sample_diffusion.py --ckpt <path_to_ckpt> --out <output.npz> ...
    ```

---

## 6. Evaluation Metrics (Planned)
We evaluate the generated `.npz` traces using:
1.  **Distribution Fidelity**: Jensen-Shannon Divergence of Event Counts and Inter-arrival times.
2.  **System Validity**:
    *   % of valid `SchedSwitch` (PID matching).
    *   % of valid File I/O cycles (Open -> Close).

---
*Created: Jan 11, 2026*
