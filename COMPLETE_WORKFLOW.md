# Complete Workflow: Step-by-Step Commands

This document provides all commands to run the complete pipeline from start to finish.

## Prerequisites

Make sure you're in the project directory and have activated your virtual environment:

```powershell
cd C:\workplace\SyntheticLogGeneration
.venv\Scripts\Activate.ps1
```

---

## Step 1: Analyze Your Dataset (Optional but Recommended)

First, understand your data before training:

```powershell
python utils/data_stats.py --data_path Datasets/compress-gzip/sequence_length_200/training
```

This will show you:
- Number of sequences
- Sequence lengths
- Event frequency distribution
- Value ranges

---

## Step 2: Train the Model

Train the SSSDS4 model on your dataset. Start with a small number of epochs to test, then do a full training run.

### Full Training Run (100 epochs - recommended):
```powershell
python train.py --dataset compress-gzip --sequence_length 200 --batch_size 32 --epochs 100 --lr 2e-4 --d_model 128 --num_layers 4 --save_dir checkpoints/compress-gzip --device cuda
```

**Note**: 
- Use `--device cuda` if you have a GPU (much faster)
- Training will take several hours on CPU
- Checkpoints are saved automatically in `checkpoints/compress-gzip/`

**What to expect:**
- Progress bars showing training progress
- Loss values decreasing over time
- Best model saved as `checkpoints/compress-gzip/best.pt`

---

## Step 3: Generate Synthetic Traces

Generate synthetic kernel traces using the trained model:

```powershell
python generate.py --checkpoint checkpoints/compress-gzip/best.pt --num_samples 100 --seq_len 200 --batch_size 16 --output_dir generated_traces/compress-gzip --output_format both --device cpu
```

**Parameters:**
- `--checkpoint`: Path to your trained model
- `--num_samples`: Number of sequences to generate (100 is good for evaluation)
- `--seq_len`: Length of each sequence (should match training length: 200)
- `--batch_size`: Batch size for generation (adjust based on memory)
- `--output_dir`: Where to save generated traces
- `--output_format`: `text`, `npy`, or `both` (both saves in both formats)
- `--device`: `cpu` or `cuda`

**What to expect:**
- Progress bar showing generation progress
- Statistics about generated sequences
- Files saved in `generated_traces/compress-gzip/`:
  - `generated_traces.txt` - Text format
  - `generated_traces.npy` - NumPy format
  - `metadata.json` - Generation metadata

---

## Step 4: Run Diagnostics (Optional but Recommended)

Check if there are any issues with the generated data:

```powershell
python diagnose_generation.py
```

This will show:
- Real data statistics
- Normalization values
- Generated data analysis
- Distribution comparison

---

## Step 5: Evaluate Generated Traces

Compare generated traces with real traces:

```powershell
python evaluate.py --generated generated_traces/compress-gzip/generated_traces.txt --real Datasets/compress-gzip/sequence_length_200/testing --output evaluation_results.json
```

**What to expect:**
- Accuracy score (how many events match exactly)
- ROUGE-L score (sequence similarity)
- Perfect rate (percentage of perfectly matched sequences)
- Distribution metrics (KL divergence, unique events, etc.)
- Results saved to `evaluation_results.json`

---

## Step 6: Analyze Results

View the evaluation results:

```powershell
# On Windows PowerShell
Get-Content evaluation_results.json | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Or just open the file in a text editor
notepad evaluation_results.json
```

---

## Complete Example: Full Pipeline

Here's the complete sequence of commands for a full run:

```powershell
# 1. Activate virtual environment
cd C:\workplace\SyntheticLogGeneration
.venv\Scripts\Activate.ps1

# 2. Analyze dataset (optional)
python utils/data_stats.py --data_path Datasets/compress-gzip/sequence_length_200/training

# 3. Train model (100 epochs)
python train.py --dataset compress-gzip --sequence_length 200 --batch_size 32 --epochs 100 --lr 2e-4 --d_model 128 --num_layers 4 --save_dir checkpoints/compress-gzip --device cpu

# 4. Generate traces
python generate.py --checkpoint checkpoints/compress-gzip/best.pt --num_samples 100 --seq_len 200 --batch_size 16 --output_dir generated_traces/compress-gzip --output_format both --device cpu

# 5. Run diagnostics
python diagnose_generation.py

# 6. Evaluate
python evaluate.py --generated generated_traces/compress-gzip/generated_traces.txt --real Datasets/compress-gzip/sequence_length_200/testing --output evaluation_results.json

# 7. View results
notepad evaluation_results.json
```

---

## Quick Test Run (For Testing)

If you just want to quickly test everything works:

```powershell
# 1. Quick training (1 epoch)
python train.py --dataset compress-gzip --sequence_length 200 --batch_size 16 --epochs 1 --lr 2e-4 --d_model 64 --num_layers 2 --save_dir checkpoints/compress-gzip_test --device cpu

# 2. Generate small sample
python generate.py --checkpoint checkpoints/compress-gzip_test/best.pt --num_samples 10 --seq_len 200 --batch_size 4 --output_dir generated_traces/test --output_format text --device cpu

# 3. Evaluate
python evaluate.py --generated generated_traces/test/generated_traces.txt --real Datasets/compress-gzip/sequence_length_200/testing --output test_results.json
```

---

## Troubleshooting

### If training is too slow:
- Reduce `--batch_size` (e.g., 16 instead of 32)
- Reduce `--d_model` (e.g., 64 instead of 128)
- Reduce `--num_layers` (e.g., 2 instead of 4)
- Use GPU if available: `--device cuda`

### If you run out of memory:
- Reduce `--batch_size`
- Reduce `--num_samples` during generation
- Use CPU instead of GPU: `--device cpu`

### If results are poor:
- Train for more epochs (200-500)
- Check `ANALYSIS_AND_NEXT_STEPS.md` for improvement suggestions
- Run diagnostics to identify issues

### If checkpoint not found:
- Make sure training completed successfully
- Check that `checkpoints/compress-gzip/best.pt` exists
- Use `latest.pt` if `best.pt` doesn't exist

---

## File Locations

After running the pipeline, you'll have:

```
SyntheticLogGeneration/
├── checkpoints/
│   └── compress-gzip/
│       ├── best.pt          # Best model checkpoint
│       ├── latest.pt        # Latest checkpoint
│       └── checkpoint_epoch_*.pt  # Periodic checkpoints
│
├── generated_traces/
│   └── compress-gzip/
│       ├── generated_traces.txt    # Generated sequences (text)
│       ├── generated_traces.npy    # Generated sequences (numpy)
│       └── metadata.json           # Generation metadata
│
└── evaluation_results.json         # Evaluation results
```

---

## Next Steps After Evaluation

1. **If results are good**: 
   - Train on other datasets
   - Experiment with hyperparameters
   - Generate larger batches

2. **If results need improvement**:
   - See `ANALYSIS_AND_NEXT_STEPS.md` for detailed improvement strategies
   - Try different model architectures
   - Train for more epochs
   - Adjust hyperparameters

3. **For research**:
   - Document results
   - Compare with baselines
   - Analyze patterns in generated sequences
   - Prepare for publication

