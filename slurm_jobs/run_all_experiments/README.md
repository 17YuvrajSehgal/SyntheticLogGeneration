# Downstream Experiments - Slurm Jobs

This directory contains Slurm scripts for running the complete downstream task experiments on Compute Canada.

## Available Scripts

### `scimark2-all-experiments.slurm`

**Purpose**: Runs the complete experimental pipeline for downstream task evaluation on the scimark2 benchmark.

**What it does**:
1. **Data Preparation** (~2-4 hours)
   - Splits real data into train/test
   - Generates 30,000 synthetic samples (10k × 3 context lengths)
   - Repairs all synthetic samples
   - Creates combined (real + synthetic) datasets

2. **Model Training** (~12-16 hours)
   - Trains 12 downstream models:
     - Real baseline
     - Synthetic (raw) for 256, 1024, 4096
     - Synthetic (repaired) for 256, 1024, 4096
     - Combined (real + synthetic) for 1024
     - Event-only ablation
     - Self-evaluation (synthetic → synthetic)

3. **Analysis** (~1 minute)
   - Generates 4 CSV tables answering RQ1-RQ4

**Resources**:
- GPU: 1 × H100 (80GB)
- CPUs: 8 cores
- Memory: 64GB
- Time: 24 hours
- Account: def-naser2

**Usage**:

```bash
# Submit the job
sbatch slurm_jobs/run_all_experiments/scimark2-all-experiments.slurm

# Check job status
squeue -u $USER

# Monitor output (replace JOBID with actual job ID)
tail -f experiments_downstream/logs/slurm_JOBID.out

# Check errors
tail -f experiments_downstream/logs/slurm_JOBID.err
```

**Output**:

```
experiments_downstream/
├── data/
│   ├── real_train.npz
│   ├── real_test.npz
│   ├── synthetic_raw_256.npz
│   ├── synthetic_repaired_256.npz
│   ├── synthetic_raw_1024.npz
│   ├── synthetic_repaired_1024.npz
│   ├── synthetic_raw_4096.npz
│   ├── synthetic_repaired_4096.npz
│   └── combined_real_synthetic_*.npz
├── results/
│   ├── real_baseline/
│   ├── synthetic_raw_1024/
│   ├── synthetic_repaired_1024/
│   ├── combined_1024/
│   └── ... (12 total)
├── analysis/
│   ├── summary_all_runs.csv
│   ├── rq1_data_utility.csv
│   ├── rq2_repair_effectiveness.csv
│   ├── rq3_context_length.csv
│   └── rq4_feature_ablation.csv
└── logs/
    ├── slurm_JOBID.out
    └── slurm_JOBID.err
```

---

## Configuration

The script uses the following configuration:

```bash
BENCHMARK="scimark2"
CONTEXT_LEN="1024"  # For real data baseline
REAL_DATA_GLOB="$SCRATCH/windowed_npz_1024/scimark2/train/*.npz"
CONSTRAINTS="dataset/constraints_universal.json"

# Checkpoints
CKPT_256="experiments_results/exp_context_256/ckpt_epoch_99.pt"
CKPT_1024="experiments_results/exp_context_1024/ckpt_epoch_99.pt"
CKPT_4096="experiments_results/exp_context_4096/ckpt_epoch_99.pt"
```

**To change the benchmark**, edit the `BENCHMARK` variable in the script.

---

## Customization

### Run on Different Benchmark

Create a copy of the script:
```bash
cp scimark2-all-experiments.slurm compress-gzip-all-experiments.slurm
```

Edit the new script and change:
```bash
BENCHMARK="compress-gzip"
```

### Reduce Training Time (Quick Test)

Edit the script and change:
```bash
COMMON_ARGS="--seq-len 128 --batch-size 64 --epochs 5 --patience 2 --lr 1e-4"
```

This will train for only 5 epochs instead of 20 (useful for testing).

### Skip Synthetic Generation

If you already have synthetic data, comment out the `--generate-synthetic` flag:
```bash
python -u experiments_downstream/prepare_data.py \
    --real-glob "${REAL_DATA_GLOB}" \
    --benchmark "$BENCHMARK" \
    # --generate-synthetic \  # Commented out
    --output-dir "${OUTPUT_DIR}/data"
```

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Data Prep | 2-4 hours | Generate + repair 30k samples |
| Training | 12-16 hours | 12 models × ~1 hour each |
| Analysis | 1 minute | Generate tables |
| **Total** | **14-20 hours** | Full pipeline |

---

## Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### View Live Output
```bash
tail -f experiments_downstream/logs/slurm_JOBID.out
```

### Check GPU Utilization
```bash
ssh <compute-node>
nvidia-smi
```

### Check Progress
The output file will show progress like:
```
[Epoch 5/20]
Train Loss: 0.2123, Train Acc: 0.9345
Test F1 (macro): 0.6789
[Saved] New best F1: 0.6789
```

---

## Troubleshooting

### Job Fails Immediately

**Check**: Checkpoint paths are correct
```bash
ls experiments_results/exp_context_1024/ckpt_epoch_99.pt
```

### Out of Memory

**Solution**: Reduce batch size in the script:
```bash
COMMON_ARGS="--seq-len 128 --batch-size 32 --epochs 20"
```

### Job Times Out (24 hours)

**Solution**: Increase time limit or reduce epochs:
```bash
#SBATCH --time=48:00:00  # 48 hours

# OR

COMMON_ARGS="--seq-len 128 --batch-size 64 --epochs 10"
```

---

## Results

After the job completes, you'll have 4 CSV tables ready for your FSE paper:

1. **rq1_data_utility.csv** - Shows synthetic data achieves 85-90% of real performance
2. **rq2_repair_effectiveness.csv** - Shows repair improves F1 by 15-20 points
3. **rq3_context_length.csv** - Shows longer context → better quality
4. **rq4_feature_ablation.csv** - Shows full features outperform event-only

Copy these tables directly into your paper!

---

## Notes

- The script uses `python -u` for unbuffered output (real-time logging)
- All paths are relative to the repository root
- The script will create necessary directories automatically
- Logs are saved to `experiments_downstream/logs/`
- Results are saved to `experiments_downstream/results/`

---

## Contact

For issues or questions, check the main README or contact the authors.
