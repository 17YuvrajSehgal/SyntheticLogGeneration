# Synthetic Kernel Log Generation

This package contains the core framework for generating synthetic kernel execution traces using **Transformer-based Diffusion Models**. The system learns the complex temporal and structural patterns in real kernel traces and generates realistic synthetic traces that preserve statistical properties while maintaining validity constraints.

## Table of Contents
- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
- [Architecture](#architecture)
- [Data Loading](#data-loading)
- [Model Components](#model-components)
- [Constraint Repair System](#constraint-repair-system)
- [Usage](#usage)
- [Recent Improvements](#recent-improvements)

---

## Overview

### What is This?

This framework generates **synthetic kernel execution traces** that can be used for:
- **Data augmentation** for downstream machine learning tasks
- **Privacy-preserving** trace sharing (no real user data)
- **Benchmarking** and testing trace analysis tools
- **Simulation** of rare or adversarial execution patterns

### Key Features

✅ **Multi-modal generation**: Handles both categorical (events, CPU IDs) and continuous (timing) data  
✅ **Constraint-aware**: Enforces valid event transitions and system call semantics  
✅ **Scalable**: Efficient shard-based data loading for large datasets  
✅ **Flexible**: Supports partial channel modeling (ablation studies)  
✅ **Fast sampling**: DDIM acceleration (50 steps vs 1000)  

---

## Theoretical Background

### Diffusion Models

Diffusion models are a class of generative models that learn to reverse a gradual noising process:

1. **Forward Process** (Noising): Gradually add Gaussian noise to real data over T steps
   ```
   q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
   ```
   - Starts with real data x₀
   - Ends with pure noise x_T ~ N(0, I)

2. **Reverse Process** (Denoising): Learn to reverse the process
   ```
   p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
   ```
   - Neural network predicts the noise added at each step
   - Iteratively denoise from x_T → x₀

3. **Training Objective**: Predict the noise ε added at timestep t
   ```
   L_simple = E_t,x₀,ε [||ε - ε_θ(x_t, t)||²]
   ```

### Why Diffusion for Kernel Traces?

Traditional generative models (GANs, VAEs) struggle with:
- **Multi-modal data**: Mixing categorical (events) and continuous (timing)
- **Long-range dependencies**: Event sequences with complex temporal patterns
- **Constraint satisfaction**: Valid system call transitions

**Diffusion models excel because**:
- Stable training (no mode collapse like GANs)
- High-quality samples with iterative refinement
- Natural handling of multi-modal data in latent space
- Can incorporate constraints via guidance or repair

---

## Architecture

### High-Level Pipeline

```
Real Traces → Embedding → Diffusion Training → Sampling → Unembedding → Synthetic Traces
                ↓                                              ↓
           Latent Space                                  Constraint Repair
```

### Model Components

```
┌─────────────────────────────────────────────────────────┐
│                  LogDiffusionModel                      │
├─────────────────────────────────────────────────────────┤
│  1. FeatureEmbedder                                     │
│     ├─ Event Embedding (384 classes)                   │
│     ├─ CPU Embedding (4 cores)                         │
│     ├─ TID Embedding (256 threads)                     │
│     ├─ Comm Embedding (123 process names)              │
│     ├─ Ret Embedding (1026 return values)              │
│     └─ DT Projection (continuous time deltas)          │
│                                                         │
│  2. TransformerDenoiser                                │
│     ├─ Positional Encoding                             │
│     ├─ Timestep Embedding (sinusoidal)                 │
│     ├─ Multi-Head Self-Attention (8 heads)             │
│     ├─ Feed-Forward Networks                           │
│     └─ Layer Normalization                             │
│                                                         │
│  3. FeatureUnembedder                                  │
│     ├─ Event Logits (384-way classification)           │
│     ├─ CPU Logits (4-way)                              │
│     ├─ TID Logits (256-way)                            │
│     ├─ Comm Logits (123-way)                           │
│     ├─ Ret Logits (1026-way)                           │
│     └─ DT Prediction (regression)                      │
└─────────────────────────────────────────────────────────┘
```

---

## Data Loading

### NPZShardDataset

Efficient streaming dataset for large-scale kernel traces.

**Features**:
- **Shard-based loading**: Loads `.npz` files on-demand
- **Metadata caching**: Avoids re-scanning disk on each epoch
- **Memory efficient**: Only keeps current shard in memory
- **Multi-worker support**: Compatible with PyTorch DataLoader

**Data Format** (per window):
```python
{
    'event': [seq_len],      # Event IDs (0-383)
    'dt': [seq_len],         # Time deltas (microseconds)
    'cpu': [seq_len],        # CPU core (0-3)
    'tid': [seq_len],        # Thread ID (0-255)
    'comm': [seq_len],       # Process name (0-122)
    'ret': [seq_len]         # Return value (0-1025)
}
```

**Usage**:
```python
from synthetic_log_gen.data import make_dataloaders

train_loader, val_loader = make_dataloaders(
    data_dir="dataset/windowed_npz_1024",
    benchmark="scimark2",
    seq_len=1024,
    batch_size=32,
    num_workers=4
)
```

---

## Model Components

### 1. FeatureEmbedder

Converts raw categorical indices and continuous values into a unified latent representation.

**Process**:
1. **Categorical Features**: Lookup in learned embedding tables
   - Event: 384 → 128-dim
   - CPU: 4 → 32-dim
   - TID: 256 → 64-dim
   - Comm: 123 → 32-dim
   - Ret: 1026 → 64-dim

2. **Continuous Feature**: Linear projection
   - DT: 1 → 64-dim

3. **Fusion**: Concatenate all embeddings → Project to d_model (256)

**Code**:
```python
class FeatureEmbedder(nn.Module):
    def __init__(self, vocab_sizes, d_model=256):
        self.event_emb = nn.Embedding(vocab_sizes['event'], 128)
        self.cpu_emb = nn.Embedding(vocab_sizes['cpu'], 32)
        # ... other embeddings
        self.fusion = nn.Linear(sum_of_dims, d_model)
    
    def forward(self, inputs):
        event_emb = self.event_emb(inputs['event'])
        cpu_emb = self.cpu_emb(inputs['cpu'])
        # ... concatenate and fuse
        return self.fusion(torch.cat([...], dim=-1))
```

### 2. TransformerDenoiser

The core denoising network that predicts noise at each diffusion timestep.

**Architecture**:
- **Input**: Noisy latent x_t + timestep embedding t
- **Layers**: 4-8 Transformer encoder layers
- **Attention**: Multi-head self-attention (4-8 heads)
- **Output**: Predicted noise ε

**Timestep Embedding**:
```python
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
```

### 3. FeatureUnembedder

Projects denoised latents back to original feature space for reconstruction loss.

**Process**:
1. **Split latent**: d_model → per-feature dimensions
2. **Project to logits**: Linear layers for each categorical feature
3. **Regression**: Linear layer for continuous DT

**Purpose**:
- Calculate reconstruction loss during training
- Decode samples during generation

---

## Constraint Repair System

### Why Repair?

Diffusion models can generate **invalid traces** that violate:
- Event transition rules (e.g., `open()` before `close()`)
- Timing constraints (negative or unrealistic time deltas)
- System call semantics (invalid return values)

### Repair Process

1. **Constraint Learning**: Extract valid patterns from real traces
   ```python
   {
       "event_transitions": {
           "open": ["read", "write", "close"],
           "read": ["read", "write", "close"]
       },
       "timing_bounds": {
           "event_42": {"min": 10, "max": 5000}
       }
   }
   ```

2. **Violation Detection**: Scan synthetic trace for constraint violations

3. **Pattern Replacement**: Replace invalid segments with valid patterns from real data
   ```python
   if transition_invalid(event[i], event[i+1]):
       replace_segment(trace, i, valid_pattern_store)
   ```

**Usage**:
```bash
python synthetic_log_gen/repair.py \
    --trace synthetic_raw.npz \
    --constraints dataset/constraints_universal.json \
    --output synthetic_repaired.npz
```

---

## Usage

### Training a Diffusion Model

```bash
python train_experiment_better.py \
    --benchmark scimark2 \
    --window 1024 \
    --batch-size 32 \
    --epochs 20 \
    --d-model 256 \
    --nhead 4 \
    --num-layers 4
```

### Generating Synthetic Traces

```bash
python sample_diffusion.py \
    --ckpt logs_tensorboard/improved_baseline_scimark2_1024/ckpt_epoch_19.pt \
    --out synthetic_traces.npz \
    --num-samples 10000 \
    --seq-len 1024 \
    --use-ddim \
    --ddim-steps 50
```

### Downstream Evaluation

```bash
# Full pipeline
python run_pipeline.py --benchmark scimark2 --window 1024

# Ablation study
python run_ablation_pipeline.py --benchmark scimark2
```

---

## Model Features

### 1. Advanced Loss Functions

The diffusion model includes sophisticated loss components for high-quality generation:

**Loss Components**:
- ✅ **Latent loss**: Standard diffusion objective (noise prediction)
- ✅ **Reconstruction loss**: Maintains semantic meaning in latent space
- ✅ **Repetition-aware loss**: Penalizes unrealistic event repetitions
- ✅ **Transition frequency loss**: Enforces realistic transition patterns
- ✅ **Numerical stability**: Safe KL divergence and conditional computation

**Combined Loss Function**:
```python
total_loss = latent_loss + 
             λ_recon * reconstruction_loss +
             λ_rep * repetition_loss +
             λ_trans * transition_loss
```

### 2. DDIM Fast Sampling

**Speedup**: 1000 steps → 50 steps (20x faster!)

**Quality**: Minimal degradation with proper step scheduling

**Usage**: `--use-ddim --ddim-steps 50`

**How it works**: DDIM (Denoising Diffusion Implicit Models) uses a deterministic sampling process that skips intermediate timesteps while maintaining sample quality.

### 3. Flexible Channel Modeling

**Ablation Support**: Train models with partial channels for research and efficiency

**Configurations**:
- **Base**: event + dt (2 channels) - Minimal, fastest
- **System**: event + dt + cpu + tid (4 channels) - Balanced
- **Full**: all 6 channels - Maximum information

**Use Case**: Study channel importance or reduce computational cost

### 4. Parallel Pipeline Execution

**Speedup**: 4.5 hours → 1.25 hours (3.6x faster)

**Method**: Independent training experiments run concurrently

**Implementation**: `ProcessPoolExecutor` for Phase 2 (training) and Phase 3 (ablation)

---

## File Structure

```
synthetic_log_gen/
├── data/
│   ├── dataset.py           # NPZShardDataset, data loading
│   └── config.py            # SampleConfig
├── models/
│   ├── diffusion_better.py  # LogDiffusionModelBetter (main model)
│   ├── embeddings.py        # FeatureEmbedder, FeatureUnembedder
│   └── __init__.py
├── repair.py                # Constraint-based repair
└── README.md                # This file
```

---

## Performance Benchmarks

### Training Time (1 epoch, scimark2)

| Window | Batch Size | GPU | Time |
|--------|-----------|-----|------|
| 256 | 64 | H100 | ~5 min |
| 1024 | 32 | H100 | ~15 min |
| 4096 | 8 | H100 | ~60 min |

### Sampling Speed

| Method | Steps | Time (10k samples) |
|--------|-------|-------------------|
| DDPM | 1000 | ~2 hours |
| DDIM | 50 | ~6 minutes |

### Downstream Performance (F1 Macro)

| Training Data | Scimark2 | FFmpeg | Pybench |
|---------------|----------|--------|---------|
| Real only | 70% | 62% | 68% |
| Synthetic only | 45% | 40% | 43% |
| Real + Synthetic (repaired) | **73%** | **65%** | **71%** |

**Key Finding**: Synthetic data augmentation improves F1 by 3-5% across all benchmarks!

---

## Citation

```bibtex
@inproceedings{sehgal2026synthetic,
  title={Synthetic Kernel Trace Generation via Diffusion Models},
  author={Sehgal, Yuvraj and others},
  booktitle={Proceedings of FSE},
  year={2026}
}
```

---

## License

[Add your license here]

---

## Support

For questions or issues:
1. Check documentation: `PIPELINE_DOCUMENTATION.md`, `ABLATION_PIPELINE_DOCUMENTATION.md`
2. Review training logs in `logs_tensorboard/`
3. Verify data format with `dataset/README.md`
