# Quick Start: Using Official SSSD Repository

## Prerequisites

1. Make sure you have the SSSD repository cloned in the `SSSD/` directory
2. Install basic dependencies (torch, numpy, tqdm)

## Quick Test Run

### Step 1: Train Model (Small Test)

```powershell
python sssd_integration/train_sssd.py \
    --dataset compress-gzip \
    --sequence_length 200 \
    --batch_size 16 \
    --n_iters 1000 \
    --iters_per_ckpt 500 \
    --iters_per_logging 100 \
    --lr 2e-4 \
    --masking rm \
    --missing_k 10 \
    --output_dir checkpoints_sssd
```

### Step 2: Generate Traces

After training, find your checkpoint directory. It will be:
```
checkpoints_sssd/compress-gzip/T200_beta00.0001_betaT0.02/
```

Then generate:

```powershell
python sssd_integration/generate_sssd.py \
    --checkpoint checkpoints_sssd/compress-gzip/T200_beta00.0001_betaT0.02 \
    --checkpoint_iter max \
    --num_samples 50 \
    --seq_len 200 \
    --batch_size 8 \
    --output_dir generated_traces_sssd/compress-gzip
```

### Step 3: Evaluate

```powershell
python evaluate.py \
    --generated generated_traces_sssd/compress-gzip/generated_traces.txt \
    --real Datasets/compress-gzip/sequence_length_200/testing \
    --output evaluation_results_sssd.json
```

## Full Training Run

For a proper training run (recommended):

```powershell
python sssd_integration/train_sssd.py \
    --dataset compress-gzip \
    --sequence_length 200 \
    --batch_size 32 \
    --n_iters 20000 \
    --iters_per_ckpt 2000 \
    --iters_per_logging 200 \
    --lr 2e-4 \
    --masking rm \
    --missing_k 10 \
    --output_dir checkpoints_sssd
```

This will take several hours. Checkpoints are saved every 2000 iterations.

## Troubleshooting

### Import Errors
Make sure SSSD/src is accessible:
```powershell
python -c "import sys; sys.path.insert(0, 'SSSD/src'); from imputers.SSSDS4Imputer import SSSDS4Imputer; print('OK')"
```

### CUDA Out of Memory
Reduce batch size:
```powershell
--batch_size 8  # or even 4
```

### Checkpoint Not Found
List available checkpoints:
```powershell
dir checkpoints_sssd\compress-gzip\T200_beta00.0001_betaT0.02\*.pkl
```

Use the iteration number directly:
```powershell
--checkpoint_iter 2000  # instead of 'max'
```

