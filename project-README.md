# Synthetic Log Generation Framework - Repository Overview

This is a **research project for generating synthetic kernel traces using Diffusion Models**, targeting an **FSE (Foundations of Software Engineering) Industrial Track paper**. The system combines deep learning with constraint-based validation and repair to ensure generated logs are both realistic and valid.

---

## 🎯 Core Objective

Generate **synthetic Linux kernel traces** that:
1. **Preserve statistical properties** of real system behavior
2. **Maintain semantic validity** (correct event transitions, timing, CPU usage)
3. **Provide hard guarantees** for industrial use (via post-hoc repair)
4. **Protect privacy** by generating synthetic data instead of sharing raw logs

---

## 🏗️ System Architecture

### 1. Data Pipeline
```
Raw Text Traces (.txt)
    ↓ [txt_to_enriched_parquet.py]
Enriched Parquet (.parquet) 
    ↓ [parquet_to_windowed_npz.py]
Windowed NPZ (.npz) - Training Data
```

**NPZ Format** contains 7 channels per event:
- `event`: Event ID (384 unique events across 9 benchmarks)
- `dt`: Log-normalized time delta: log(1 + Δt)
- `cpu`: CPU core ID (0-3)
- `tid`: Thread ID (hashed to 256 buckets)
- `fd`: File descriptor (capped at 1024)
- `comm`: Command/process name (tokenized)
- `ret`: Return value (tokenized)

---

### 2. Diffusion Model (`LogDiffusionModel`)

**Architecture:**
- **Transformer-based DDPM** (Denoising Diffusion Probabilistic Model)
- **Embedder**: Converts discrete tokens → continuous latent space (d_model=512)
- **Denoiser**: 8-layer Transformer with 8 attention heads
- **Head**: Projects latents back to discrete predictions

**Training:**
- **Loss**: Hybrid = Latent MSE + 0.1 × Reconstruction Loss
- **Optimizations**: BFloat16 mixed precision, TF32, torch.compile
- **Hardware**: H100 GPUs (80GB)
- **Timesteps**: 1000 diffusion steps

**Generation:**
- Starts with Gaussian noise
- Iteratively denoises over 1000 steps
- Outputs discrete event sequences

---

### 3. Constraint System (Post-Hoc Repair)

**Learning Phase** (`learn_constraints.py`):
Mines real traces to extract:
1. **Transition Graph**: Valid event sequences (E_t → E_{t+1})
2. **Temporal Bounds**: Min/max timing per event
3. **CPU Affinity**: Which CPUs trigger which events
4. **Thread/FD/Comm/Ret Constraints**: Valid combinations

**Validation** (`validate.py`):
Scores synthetic traces on:
- Transition validity
- Timing validity
- CPU consistency (global + per-event)

**Repair** (`repair.py`):
- **Greedy forward repair** of invalid transitions
- **Probabilistic sampling** from learned distributions
- **Guarantees 100% logical validity** post-repair

---

## 📊 Experimental Design

### Experiment 1: Feature Ablation (seq_len=1024)
| Config | Channels | Purpose |
|--------|----------|---------|
| `ablation_base` | event, dt | Baseline (text-only) |
| `ablation_system` | +cpu, tid | System-aware |
| `ablation_full` | +comm, ret | Full metadata |

### Experiment 2: Context Length Study
| Config | Seq Length | Batch Size | Total Tokens |
|--------|------------|------------|--------------|
| `context_256` | 256 | 256 | 65k |
| `context_1024` | 1024 | 128 | 131k |
| `context_4096` | 4096 | 32 | 131k |

---

## 🔧 Key Scripts

### Training
- `train_experiment.py`: Main training loop with H100 optimizations
- Outputs: Checkpoints + TensorBoard logs

### Generation
- `sample_diffusion.py`: Inference from trained checkpoints
- Outputs: `.npz` files with synthetic traces

### Constraint Pipeline
- `learn_constraints.py`: Extract rules from real data
- `validate.py`: Score synthetic traces
- `repair.py`: Fix constraint violations
- `run_repair_pipeline.py`: End-to-end validation → repair → re-validation

### Evaluation
- `robust_eval.py`: Comprehensive metrics (JS divergence, entropy, n-gram overlap, etc.)
- `transition_eval.py`: Transition-specific analysis
- `privacy_analysis.py`: Memorization detection

---

## 📁 Directory Structure

```
SyntheticLogGeneration/
├── dataset/
│   ├── window_shards/          # Organized NPZ training data
│   ├── metadata_all_events/    # Vocabularies (vocab.json, vocab_comm.json, etc.)
│   └── constraints_universal.json  # Learned constraints (106MB!)
├── synthetic_log_gen/          # Core package
│   ├── models/diffusion.py     # LogDiffusionModel
│   ├── data/dataset.py         # NPZShardDataset
│   ├── validate.py             # Constraint validation
│   └── repair.py               # Post-hoc repair
├── experiments_results/        # Training outputs
│   ├── exp_context_256/
│   ├── exp_context_1024/
│   └── exp_context_4096/
├── generated_traces/           # Synthetic outputs
├── repaired_traces/            # Post-repair outputs
├── slurm_jobs/                 # Cluster job scripts
└── evaluation_metrics/         # Analysis scripts
```

---

## 🎓 Research Contributions

1. **First diffusion model for kernel traces** (vs RNNs/GANs)
2. **Multi-modal learning** (7 channels vs event-only)
3. **Constraint-guided repair** for hard validity guarantees
4. **Scalability** to 4096-token contexts
5. **Industrial applicability** (privacy-preserving, guaranteed validity)

---

## 🚀 Current Status

Based on conversation history:
- ✅ Models trained for all 3 context lengths (256, 1024, 4096)
- ✅ Constraint learning completed (universal constraints extracted)
- ✅ Repair system implemented and validated
- 🔄 **Current focus**: Generating samples from trained models for evaluation

---

## 📖 Additional Documentation

- [`README.md`](README.md): Core training and sampling workflow
- [`README_EXPERIMENTS.md`](README_EXPERIMENTS.md): Detailed experimental configurations
- [`README_learn_constraints.md`](README_learn_constraints.md): Constraint learning system
- [`fse_evaluation_report_template.md`](fse_evaluation_report_template.md): FSE paper evaluation template
- [`dataset/README_npz.md`](dataset/README_npz.md): NPZ data format specification
- [`dataset/README_parquet.md`](dataset/README_parquet.md): Parquet enrichment details

---

This is a **well-structured, production-ready research codebase** with clear separation between data processing, model training, constraint learning, and evaluation. The system is designed for reproducibility and scalability on HPC clusters (Compute Canada).
