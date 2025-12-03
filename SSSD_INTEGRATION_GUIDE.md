# SSSD Official Repository Integration Guide

This guide explains how to use the official SSSD repository for training and generating synthetic kernel traces.

## Overview

We've integrated the official SSSD (Structured State Space Diffusion) repository into our project. This provides:
- ✅ Official, tested model implementations
- ✅ Optimized S4 layers
- ✅ Proven training procedures
- ✅ Better performance than our custom implementation

## Setup

### 1. Install SSSD Dependencies

The official SSSD repository has its own requirements. Install them:

```powershell
cd SSSD/src
pip install -r requirements.txt
cd ../..
```

**Note**: The requirements.txt is very large. You may want to install only the essential packages:
- torch
- numpy
- tqdm

### 2. Verify Integration

Check that the integration scripts can import SSSD modules:

```powershell
python -c "import sys; sys.path.insert(0, 'SSSD/src'); from imputers.SSSDS4Imputer import SSSDS4Imputer; print('SSSD import successful!')"
```

## Usage

### Training with Official SSSD

Use the integrated training script that adapts SSSD to our data format:

```powershell
python sssd_integration/train_sssd.py \
    --dataset compress-gzip \
    --sequence_length 200 \
    --batch_size 32 \
    --n_iters 10000 \
    --iters_per_ckpt 1000 \
    --iters_per_logging 100 \
    --lr 2e-4 \
    --masking rm \
    --missing_k 10 \
    --output_dir checkpoints_sssd
```

**Parameters:**
- `--dataset`: Dataset name (e.g., compress-gzip)
- `--sequence_length`: Length of sequences (200)
- `--batch_size`: Batch size (32)
- `--n_iters`: Number of training iterations (10000+)
- `--iters_per_ckpt`: Save checkpoint every N iterations (1000)
- `--iters_per_logging`: Log every N iterations (100)
- `--lr`: Learning rate (2e-4)
- `--masking`: Masking strategy - 'rm' (random), 'bm' (blackout), 'mnr' (missing not at random)
- `--missing_k`: Number of missing points for masking
- `--output_dir`: Where to save checkpoints

### Generation with Official SSSD

Generate synthetic traces using the trained model:

```powershell
python sssd_integration/generate_sssd.py \
    --checkpoint checkpoints_sssd/compress-gzip/T200_beta00.0001_betaT0.02 \
    --checkpoint_iter max \
    --num_samples 100 \
    --seq_len 200 \
    --batch_size 16 \
    --output_dir generated_traces_sssd/compress-gzip
```

**Parameters:**
- `--checkpoint`: Path to checkpoint directory (contains .pkl files)
- `--checkpoint_iter`: Which checkpoint to load ('max' for latest, or specific number)
- `--num_samples`: Number of sequences to generate
- `--seq_len`: Length of each sequence
- `--batch_size`: Batch size for generation
- `--output_dir`: Where to save generated traces

### Evaluation

Use the same evaluation script as before:

```powershell
python evaluate.py \
    --generated generated_traces_sssd/compress-gzip/generated_traces.txt \
    --real Datasets/compress-gzip/sequence_length_200/testing \
    --output evaluation_results_sssd.json
```

## Key Differences from Custom Implementation

### 1. Data Format
- **Official SSSD**: Expects `(batch, channels, seq_len)` format
- **Our Data**: `(batch, seq_len)` format
- **Solution**: We reshape data in the integration layer

### 2. Training Approach
- **Official SSSD**: Trains on imputation tasks (filling missing values)
- **Our Goal**: Unconditional generation
- **Solution**: We use masking during training, then generate from pure noise

### 3. Model Architecture
- **Official SSSD**: More sophisticated S4 implementation
- **Our Custom**: Simplified version
- **Benefit**: Better performance with official implementation

## Complete Workflow

### Step 1: Train Model
```powershell
python sssd_integration/train_sssd.py \
    --dataset compress-gzip \
    --sequence_length 200 \
    --batch_size 32 \
    --n_iters 20000 \
    --iters_per_ckpt 2000 \
    --lr 2e-4 \
    --output_dir checkpoints_sssd
```

### Step 2: Generate Traces
```powershell
python sssd_integration/generate_sssd.py \
    --checkpoint checkpoints_sssd/compress-gzip/T200_beta00.0001_betaT0.02 \
    --num_samples 100 \
    --seq_len 200 \
    --output_dir generated_traces_sssd/compress-gzip
```

### Step 3: Evaluate
```powershell
python evaluate.py \
    --generated generated_traces_sssd/compress-gzip/generated_traces.txt \
    --real Datasets/compress-gzip/sequence_length_200/testing \
    --output evaluation_results_sssd.json
```

## File Structure

```
SyntheticLogGeneration/
├── SSSD/                          # Official repository (cloned)
│   └── src/
│       ├── imputers/
│       ├── utils/
│       └── ...
├── sssd_integration/              # Our integration layer
│   ├── train_sssd.py             # Training script
│   ├── generate_sssd.py          # Generation script
│   └── __init__.py
├── checkpoints_sssd/              # Trained models
│   └── compress-gzip/
│       └── T200_beta00.0001_betaT0.02/
│           ├── 1000.pkl
│           ├── 2000.pkl
│           └── normalization.json
└── generated_traces_sssd/         # Generated sequences
    └── compress-gzip/
        ├── generated_traces.txt
        ├── generated_traces.npy
        └── metadata.json
```

## Advantages of Using Official SSSD

1. **Better Performance**: Optimized S4 layers and proven architecture
2. **More Robust**: Tested on multiple datasets
3. **Active Maintenance**: Official repository with updates
4. **Research Proven**: Used in published papers

## Troubleshooting

### Import Errors
If you get import errors, make sure SSSD/src is in the Python path:
```python
import sys
sys.path.insert(0, 'SSSD/src')
```

### CUDA Out of Memory
- Reduce `--batch_size`
- Reduce model size in `train_sssd.py` (res_channels, num_res_layers)

### Checkpoint Not Found
- Check the checkpoint directory path
- Use `--checkpoint_iter max` to find the latest checkpoint
- Verify checkpoint files have `.pkl` extension

### Generation Quality Issues
- Train for more iterations (`--n_iters 50000+`)
- Try different masking strategies
- Adjust learning rate
- Check normalization values are correct

## Next Steps

1. **Train on Multiple Datasets**: Test on all 9 datasets
2. **Hyperparameter Tuning**: Experiment with model sizes and training parameters
3. **Compare Results**: Compare official SSSD vs custom implementation
4. **Research Improvements**: Build on top of official implementation

## References

- Official SSSD Repository: [GitHub](https://github.com/AI4HealthUOL/SSSD)
- Paper: "Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models" (TMLR 2022)

