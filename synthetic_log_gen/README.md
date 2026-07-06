# TraceSynth — Diffusion Model & Constraint Repair

This package implements the **Transformer-based diffusion model** at the core of **TraceSynth** (Stage 2), together with the constraint validation and repair modules (Stage 3). The model learns complex temporal and structural patterns from real LTTng kernel traces and generates realistic synthetic traces that preserve statistical properties while maintaining validity constraints.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Module Reference](#module-reference)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Overview

### Purpose

Generate **synthetic kernel execution traces** for:
- **Data augmentation** for downstream ML tasks (anomaly detection, workload classification)
- **Privacy-preserving** trace sharing (no real user data exposure)
- **Benchmarking** and testing trace analysis tools
- **Simulation** of rare or adversarial execution patterns

### Key Features

- ✅ **Multi-channel modeling**: The six channels reported in the paper (event, dt, cpu, tid, comm, ret); the code additionally supports an `fd` channel present in the NPZ shards
- ✅ **Constraint-aware generation**: Enforces valid event transitions and system call semantics
- ✅ **Scalable data loading**: Efficient shard-based streaming for large datasets
- ✅ **Fast sampling**: DDIM acceleration (50 steps vs 1000 steps, 20x faster)
- ✅ **Validation and repair**: Post-generation constraint checking and fixing

---

## Architecture

### Pipeline Overview

```mermaid
graph LR
    A["Real Traces (NPZ)"] --> B["FeatureEmbedder"]
    B --> C["Latent Space (d_model)"]
    C --> D["TransformerDenoiser"]
    D --> E["Diffusion Training"]
    E --> F["Trained Model"]
    F --> G["DDIM Sampling"]
    G --> H["FeatureUnembedder"]
    H --> I["Synthetic Traces"]
    I --> J["Constraint Validation"]
    J --> K["Repair (if needed)"]
    K --> L["Valid Synthetic Traces"]
```

### Diffusion Process

**Forward Process** (Training): Add Gaussian noise to real data over T timesteps
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**Reverse Process** (Sampling): Learn to denoise from pure noise to data
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Training Objective**: Predict the noise ε added at timestep t
```
L = E_t,x₀,ε [||ε - ε_θ(x_t, t)||²] + λ_recon * L_recon + λ_rep * L_rep + λ_trans * L_trans
```

---

## Module Reference

### 1. Data Loading (`data/`)

#### `NPZShardDataset`
Efficient streaming dataset for large-scale kernel traces stored as NPZ shards.

**Features**:
- Shard-based loading with LRU caching
- Memory-efficient (loads shards on-demand)
- Multi-worker DataLoader compatible
- Configurable channel selection for ablation studies

**Data Format** (per sample):
```python
{
    'event': [seq_len],  # Event IDs (int32)
    'dt': [seq_len],     # Log-normalized time deltas (float32)
    'cpu': [seq_len],    # CPU core ID (int8)
    'tid': [seq_len],    # Thread ID bucket (int16)
    'fd': [seq_len],     # File descriptor (int16)
    'comm': [seq_len],   # Process name ID (int16)
    'ret': [seq_len]     # Return value ID (int16)
}
```

**Usage**:
```python
from synthetic_log_gen.data import NPZShardDataset, SampleConfig

config = SampleConfig(
    seq_len=1024,
    channels=("event", "dt", "cpu", "tid", "comm", "ret")
)

dataset = NPZShardDataset(
    shard_paths=["path/to/shard1.npz", "path/to/shard2.npz"],
    config=config,
    cache_shards=2
)
```

#### `SampleConfig`
Configuration dataclass for dataset loading.

**Parameters**:
- `seq_len`: Sequence length (default: 1024)
- `channels`: Tuple of channel names to load (default: all 7 channels)
- `return_dict`: Return dict instead of tensor (default: False)
- `dtype_*`: Data types for each channel

**Helper Method**:
- `get_dim()`: Returns total embedding dimension for configured channels

---

### 2. Model Components (`models/`)

#### `FeatureEmbedder`
Converts raw categorical indices and continuous values into unified latent representation.

**Process**:
1. **Discrete features**: Embedding lookup (event, cpu, tid, fd, comm, ret)
2. **Continuous feature**: MLP projection (dt)
3. **Fusion**: Sum all embeddings → `[B, L, d_model]`

**Implementation**:
```python
class FeatureEmbedder(nn.Module):
    def __init__(self, d_model, vocab_sizes, dropout=0.1):
        # Creates nn.Embedding for each discrete channel
        # Creates MLP for continuous dt channel
        
    def forward(self, inputs: dict) -> torch.Tensor:
        # Returns: [B, L, d_model]
```

#### `TransformerDenoiser`
Core denoising network that predicts noise at each diffusion timestep.

**Architecture**:
- **Input**: Noisy latent `x_t` + sinusoidal timestep embedding `t`
- **Layers**: 4-8 Transformer encoder layers
- **Attention**: Multi-head self-attention (4-8 heads)
- **Output**: Predicted noise `ε`

**Timestep Embedding**: Sinusoidal positional encoding for diffusion timestep

#### `FeatureUnembedder`
Projects denoised latents back to original feature space.

**Process**:
1. **Discrete channels**: Linear projection to logits (classification)
2. **Continuous channel (dt)**: Linear projection to scalar (regression)

**Output**: Dictionary of logits/predictions for each channel

#### `LogDiffusionModelBetter`
Complete diffusion model with advanced training features.

**Key Components**:
- `FeatureEmbedder`: Input embedding
- `TransformerDenoiser`: Noise prediction network
- `FeatureUnembedder`: Output projection
- Noise scheduler (linear beta schedule)

**Advanced Loss Functions**:
1. **Latent loss**: Standard diffusion objective (MSE on predicted noise)
2. **Reconstruction loss**: Maintains semantic meaning via cross-entropy/MSE
3. **Repetition loss**: Penalizes unrealistic event repetition patterns
4. **Transition frequency loss**: Enforces realistic transition distributions

**Sampling Methods**:
- `sample()`: Standard DDPM sampling (1000 steps)
- `sample_ddim()`: Fast DDIM sampling (50 steps, 20x faster)

**Configuration**:
```python
model = LogDiffusionModelBetter(
    vocab_sizes={
        'event': 384,
        'cpu': 4,
        'tid': 256,
        'fd': 1025,
        'comm': 123,
        'ret': 1026
    },
    d_model=256,
    nhead=8,
    num_layers=4,
    max_timesteps=1000,
    dropout=0.1,
    target_repeats=279,        # From real data statistics
    repetition_weight=0.05,
    transition_weight=0.03
)
```

---

### 3. Constraint System

#### `validate.py`
Validates generated traces against learned constraints.

**Validation Checks**:
1. **Event transitions**: Checks if `event[t] → event[t+1]` is valid
2. **Timing bounds**: Verifies `dt` values are within learned min/max
3. **CPU affinity (global)**: Checks if CPU ID exists in system
4. **CPU affinity (local)**: Checks if event can occur on specific CPU

**Usage**:
```bash
python synthetic_log_gen/validate.py \
    --trace synthetic_output.npz \
    --constraints dataset/constraints_universal.json \
    --output validity_report.json
```

**Output Format**:
```json
{
  "validity_score": {
    "transitions": 95.2,
    "timing": 98.7,
    "cpu_global": 100.0,
    "cpu_local": 94.3
  },
  "details": {
    "total_samples": 10000,
    "seq_len": 1024,
    "invalid_transitions": 49152,
    "invalid_dts": 13312
  }
}
```

#### `repair.py`
Repairs constraint violations in generated traces using probabilistic replacement.

**Repair Strategy**:
1. **Transition repair**: Replace invalid `event[t+1]` with valid successor sampled from learned probabilities
2. **Timing repair**: Adjust `dt` values to match constraints for new events
3. **CPU repair**: Move events to allowed CPUs

**Usage**:
```bash
python synthetic_log_gen/repair.py \
    --trace synthetic_raw.npz \
    --constraints dataset/constraints_universal.json \
    --output synthetic_repaired.npz
```

**Repair Process**:
- Scans each trace for violations
- Replaces invalid segments with valid patterns
- Uses learned event probabilities for realistic sampling
- Reports repair statistics (repairs made / total checks)

---

## Usage

### Training a Diffusion Model

```bash
python train_experiment.py \
    --data-root scratch/windowed_npz_1024 \
    --benchmark scimark2 \
    --seq-len 1024 \
    --batch-size 32 \
    --epochs 20 \
    --d-model 256 \
    --nhead 8 \
    --num-layers 4 \
    --lr 2e-4
```

**Key Arguments**:
- `--data-root`: Root directory containing the windowed NPZ `train/val/test` folders
- `--benchmark`: Dataset benchmark subdirectory (e.g., `scimark2`, `ffmpeg`, `pybench`)
- `--seq-len`: Sequence length / context length (256, 1024, or 4096)
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs
- `--d-model`: Transformer hidden dimension
- `--nhead`: Number of attention heads
- `--num-layers`: Number of Transformer layers

**Output**: Checkpoints saved to `logs_tensorboard/<experiment_name>/`

### Generating Synthetic Traces

```bash
python sample_diffusion.py \
    --ckpt logs_tensorboard/experiment_name/ckpt_epoch_19.pt \
    --out synthetic_traces.npz \
    --num-samples 10000 \
    --seq-len 1024 \
    --use-ddim \
    --ddim-steps 50
```

**Key Arguments**:
- `--ckpt`: Path to trained model checkpoint
- `--out`: Output NPZ file path
- `--num-samples`: Number of traces to generate
- `--seq-len`: Sequence length (must match training)
- `--use-ddim`: Enable fast DDIM sampling
- `--ddim-steps`: Number of DDIM steps (default: 50)

### Validating Generated Traces

```bash
python synthetic_log_gen/validate.py \
    --trace synthetic_traces.npz \
    --constraints dataset/constraints_universal.json \
    --output validity_report.json
```

### Repairing Invalid Traces

```bash
python synthetic_log_gen/repair.py \
    --trace synthetic_traces.npz \
    --constraints dataset/constraints_universal.json \
    --output synthetic_repaired.npz
```

---

## File Structure

```
synthetic_log_gen/
├── data/
│   ├── __init__.py          # Exports: NPZShardDataset, SampleConfig
│   └── dataset.py           # Dataset implementation with shard loading
├── models/
│   ├── __init__.py          # Exports: LogDiffusionModelBetter, embeddings
│   ├── diffusion.py         # Original diffusion model (legacy)
│   ├── diffusion_better.py  # Improved model with advanced losses
│   └── embeddings.py        # FeatureEmbedder, FeatureUnembedder
├── validate.py              # Constraint validation script
├── repair.py                # Constraint repair script
└── README.md                # This file
```

---

## Model Specifications

### Default Vocabulary Sizes

Based on the LTTng Phoronix dataset:

| Channel | Vocabulary Size | Description |
|---------|----------------|-------------|
| `event` | 384 | Unique kernel event types |
| `cpu` | 4 | CPU cores (0-3) |
| `tid` | 256 | Thread ID buckets (hashed) |
| `fd` | 1025 | File descriptors (0-1024, clamped) |
| `comm` | 123 | Process command names |
| `ret` | 1026 | Return values (Top-K + special tokens) |
| `dt` | 1 | Continuous time delta (log-normalized) |

### Training Configuration

**Recommended Settings** (1024 sequence length):
- `d_model`: 256
- `nhead`: 8
- `num_layers`: 4
- `batch_size`: 32
- `learning_rate`: 1e-4
- `max_timesteps`: 1000
- `dropout`: 0.1

**Training Time** (H100 GPU):
- 256 window: ~5 min/epoch
- 1024 window: ~15 min/epoch
- 4096 window: ~60 min/epoch

### Sampling Performance

| Method | Steps | Time (10K samples) | Quality |
|--------|-------|-------------------|---------|
| DDPM | 1000 | ~2 hours | Highest |
| DDIM | 50 | ~6 minutes | Very Good (minimal degradation) |

---

## Advanced Features

### Flexible Channel Modeling

Train models with partial channels for ablation studies:

```python
config = SampleConfig(
    seq_len=1024,
    channels=("event", "dt")  # Minimal configuration
)
```

**Common Configurations** (the feature-richness settings used in the RQ4 ablation):
- **Base**: `("event", "dt")` - 2 channels, fastest training
- **System**: `("event", "dt", "cpu", "tid")` - 4 channels, balanced
- **Full**: `("event", "dt", "cpu", "tid", "comm", "ret")` - 6 channels, maximum information

### Constraint Learning

Constraints are learned from real data using `data_processing/learn_constraints.py`:

```bash
python data_processing/learn_constraints.py \
    --real_glob "dataset/windowed_npz_1024/**/*.npz" \
    --output dataset/constraints_universal.json \
    --num_events 384
```

**Learned Invariants**:
1. **Transition graph**: Valid event sequences
2. **Temporal bounds**: Min/max time deltas per event
3. **CPU affinity**: Allowed CPUs per event
4. **Thread identity**: Allowed TID buckets per event
5. **Semantic context**: Allowed comm, fd, ret values per event

---

## Notes

### Data Format Compatibility

- Input NPZ files must match the format produced by `data_processing/parquet_to_windowed_npz.py`
- All channels use the same sequence length `L`
- Time deltas (`dt`) are log-normalized: `log(1 + Δt)`
- Thread IDs are hashed: `tid % 256`
- File descriptors are clamped: `fd ≥ 1024 → 1024`

### Memory Management

- `NPZShardDataset` uses LRU caching (default: 2 shards)
- Increase `cache_shards` for faster training if memory allows
- Use `num_workers > 0` in DataLoader for parallel loading
- Set `prefetch_factor` to pre-load batches (default: 2)

### Best Practices

1. **Always validate** generated traces before use
2. **Repair violations** for downstream tasks requiring hard guarantees
3. **Use DDIM** for fast sampling unless highest quality is critical
4. **Monitor TensorBoard** logs during training for convergence
5. **Match sequence length** between training and sampling
