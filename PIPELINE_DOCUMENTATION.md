# Downstream Evaluation Pipeline Documentation

## Overview

The **Downstream Evaluation Pipeline** (`run_pipeline.py`) is an automated workflow that evaluates the quality and utility of synthetic kernel trace data by training next-event prediction models on different data configurations. This pipeline answers the research question: **"Does synthetic data improve downstream task performance?"**

## Purpose

This pipeline automates the complete experimental workflow for evaluating synthetic log generation quality through downstream utility. It:

1. Generates synthetic kernel traces using trained diffusion models
2. Applies constraint-based repair to ensure validity
3. Trains next-event prediction models on various data configurations
4. Compares performance to establish synthetic data utility

## Research Questions Addressed

- **RQ1**: How does synthetic data quality compare to real data for downstream tasks?
- **RQ2**: Does data augmentation (real + synthetic) improve model performance?
- **RQ3**: Is constraint-guided repair necessary for synthetic data utility?

---

## Pipeline Architecture

### Execution Phases

The pipeline executes in **2 phases**:

#### **Phase 1: Data Preparation** (Sequential - Steps 1-5)
Sequential execution ensures data dependencies are met.

#### **Phase 2: Model Training** (Parallel - Steps 6-9)
Parallel execution for efficiency (4 independent experiments).

---

## Pipeline Steps

### Step 1: Prepare Real Data
**Purpose**: Split real kernel traces into training and test sets

**Input**:
- Raw windowed traces: `$SCRATCH/window_shards/windowed_npz_{window}/{benchmark}/train/*.npz`

**Output**:
- `real_train.npz` - Training set (80%)
- `real_test.npz` - Test set (20%)

**Channels**: event, dt, cpu, tid, comm, ret (6 channels)

---

### Step 2: Generate Synthetic Data
**Purpose**: Generate synthetic traces using trained diffusion model

**Input**:
- Trained diffusion checkpoint: `logs_tensorboard/improved_baseline_{benchmark}_{window}/ckpt_epoch_{N}.pt`

**Output**:
- `synthetic_raw_{window}_{N}k.npz` - Raw synthetic traces (10,000 samples by default)

**Model Configuration**:
- Architecture: Transformer-based diffusion model
- d_model: 256
- nhead: 4
- num_layers: 4
- Sampling: DDIM (50 steps)

**Channels Generated**: Same 6 channels as real data

---

### Step 3: Create Combined Dataset (No Repair)
**Purpose**: Create hybrid dataset without constraint repair

**Input**:
- `real_train.npz`
- `synthetic_raw_{window}_{N}k.npz`

**Output**:
- `combined_real_synthetic_norepair_{window}_50_50.npz`

**Mixing Ratio**: 50% real + 50% synthetic

---

### Step 4: Repair Synthetic Data
**Purpose**: Apply constraint-based repair to fix violations

**Input**:
- `synthetic_raw_{window}_{N}k.npz`
- `dataset/constraints_universal.json` - Universal constraints

**Output**:
- `synthetic_repaired_{window}_{N}k.npz`

**Constraints Enforced**:
- Valid event transitions
- Timing bounds (dt ranges)
- CPU affinity rules
- System call semantics

**Repair Method**: Pattern-based replacement using real trace segments

---

### Step 5: Create Combined Dataset (Repaired)
**Purpose**: Create hybrid dataset with repaired synthetic data

**Input**:
- `real_train.npz`
- `synthetic_repaired_{window}_{N}k.npz`

**Output**:
- `combined_real_synthetic_repaired_{window}_50_50.npz`

**Mixing Ratio**: 50% real + 50% repaired synthetic

---

### Step 6: Train on Real Data (Baseline)
**Purpose**: Establish baseline performance with real data only

**Training Data**: `real_train.npz`
**Test Data**: `real_test.npz`

**Model**: NextEventPredictor (Transformer-based)
- Task: Predict next event given sequence
- Input: 128-token sequences
- Output: 384 event classes

**Training**:
- Epochs: 20
- Batch size: 64 (256/1024) or 32 (4096)
- Optimizer: AdamW
- Learning rate: 1e-4

---

### Step 7: Train on Synthetic Only
**Purpose**: Evaluate synthetic data quality in isolation

**Training Data**: `synthetic_raw_{window}_{N}k.npz` (synthetic only)
**Test Data**: `real_test.npz` (always real)

**Expected Result**: Lower performance than baseline (synthetic ≠ real)

---

### Step 8: Train on Combined (Repaired)
**Purpose**: Evaluate data augmentation with repaired synthetic data

**Training Data**: `combined_real_synthetic_repaired_{window}_50_50.npz`
**Test Data**: `real_test.npz`

**Hypothesis**: Repaired synthetic + real > real alone (data augmentation benefit)

---

### Step 9: Train on Combined (No Repair)
**Purpose**: Evaluate necessity of constraint repair

**Training Data**: `combined_real_synthetic_norepair_{window}_50_50.npz`
**Test Data**: `real_test.npz`

**Hypothesis**: Repaired > No repair (constraint violations hurt performance)

---

## Output Metrics

Each training experiment produces comprehensive metrics:

### Primary Metrics
- **F1 Score (Macro)**: Average F1 across all event classes (main metric)
- **F1 Score (Weighted)**: Weighted by class frequency
- **Accuracy**: Overall prediction accuracy

### Secondary Metrics
- **Top-5 Accuracy**: Correct event in top 5 predictions
- **Top-10 Accuracy**: Correct event in top 10 predictions
- **Loss**: Cross-entropy loss

### Output Files (per experiment)
```
experiments_downstream_results/{benchmark}/{window}/results/{run_name}/
├── config.json           # Training configuration
├── final_metrics.json    # Final test metrics
├── history.json          # Training history (all epochs)
├── best_model.pt         # Best model checkpoint
└── predictions.npz       # Model predictions on test set
```

---

## Expected Results Pattern

| Training Data | F1 (Macro) | Interpretation |
|---------------|------------|----------------|
| Real only (baseline) | ~70% | Upper bound (real data) |
| Synthetic only | ~40-50% | Synthetic quality check |
| Combined (repaired) | ~72-75% | **Data augmentation benefit** ✅ |
| Combined (no repair) | ~65-68% | Constraint violations hurt |

**Key Finding**: Repaired synthetic + real > real alone → Synthetic data provides utility

---

## Usage

### Basic Usage
```bash
python run_pipeline.py --benchmark scimark2 --window 256
```

### Advanced Usage
```bash
python run_pipeline.py \
    --benchmark ffmpeg \
    --window 1024 \
    --checkpoint-epoch 19 \
    --num-samples 10000 \
    --skip-steps 1 2 3
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--benchmark` | Benchmark name (scimark2, pybench, ffmpeg, etc.) | Required |
| `--window` | Sequence length (256, 1024, 4096) | Required |
| `--checkpoint-epoch` | Diffusion model epoch to use | 19 |
| `--num-samples` | Number of synthetic samples to generate | 10000 |
| `--skip-steps` | Steps to skip (e.g., `1 2 3` to skip generation) | None |
| `--scratch` | Scratch directory path | `$SCRATCH` |

---

## Cluster Submission

### Generate Job Scripts
```bash
python generate_pipeline_jobs.py
```

This creates Slurm scripts for all benchmark/window combinations:
```
slurm_jobs/pipelines/
├── run_scimark2_256.slurm
├── run_scimark2_1024.slurm
├── run_scimark2_4096.slurm
├── run_ffmpeg_256.slurm
└── ...
```

### Submit Jobs
```bash
# Single job
sbatch slurm_jobs/pipelines/run_scimark2_256.slurm

# All jobs
for f in slurm_jobs/pipelines/*.slurm; do sbatch $f; done
```

---

## Time Estimates

| Window Size | Phase 1 (Sequential) | Phase 2 (Parallel) | Total |
|-------------|---------------------|-------------------|-------|
| 256 | ~15 min | ~30 min | ~45 min |
| 1024 | ~30 min | ~45 min | ~1.25 hr |
| 4096 | ~2 hr | ~2 hr | ~4 hr |

**Note**: Phase 2 runs 4 experiments in parallel, saving ~3x time vs sequential

---

## Directory Structure

```
experiments_downstream_results/{benchmark}/{window}/
├── data/
│   ├── real_train.npz                              # Step 1
│   ├── real_test.npz                               # Step 1
│   ├── synthetic_raw_{window}_{N}k.npz            # Step 2
│   ├── combined_real_synthetic_norepair_*.npz     # Step 3
│   ├── synthetic_repaired_{window}_{N}k.npz       # Step 4
│   └── combined_real_synthetic_repaired_*.npz     # Step 5
└── results/
    ├── real_baseline_{benchmark}_{window}/         # Step 6
    ├── synthetic_data_only_{benchmark}_{window}/   # Step 7
    ├── combined_50_50_{benchmark}_{window}/        # Step 8
    └── combined_50_50_norepair_{benchmark}_{window}/ # Step 9
```

---

## Troubleshooting

### Common Issues

**Issue**: Checkpoint not found
```
FileNotFoundError: Checkpoint not found: .../ckpt_epoch_19.pt
```
**Solution**: Verify checkpoint exists or specify different epoch:
```bash
--checkpoint-epoch 18
```

**Issue**: OOM during generation (4096 window)
```
RuntimeError: CUDA out of memory
```
**Solution**: Batch size auto-adjusts, but may need smaller `--num-samples`

**Issue**: Step failed, want to resume
```bash
# Skip completed steps
python run_pipeline.py --benchmark scimark2 --window 256 --skip-steps 1 2 3
```

---

## Performance Optimization

### Batch Size Auto-Tuning
The pipeline automatically adjusts batch sizes based on window size:

| Window | Sample Batch | Train Batch |
|--------|-------------|-------------|
| 256 | 64 | 64 |
| 1024 | 32 | 64 |
| 4096 | 8 | 32 |

### Parallel Execution
Steps 6-9 run in parallel with `ProcessPoolExecutor(max_workers=4)`, utilizing multiple CPU cores efficiently.

---

## Related Scripts

- **`run_ablation_pipeline.py`**: Ablation study variant (channel importance)
- **`analyze_results.py`**: Aggregate and visualize results
- **`collect_ablation_results.py`**: Collect ablation study metrics

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{sehgal2026synthetic,
  title={Synthetic Kernel Trace Generation via Diffusion Models},
  author={Sehgal, Yuvraj and others},
  booktitle={Proceedings of FSE},
  year={2026}
}
```

---

## Support

For issues or questions:
1. Check logs in `$SCRATCH/pipeline_{benchmark}_{window}_{jobid}.out`
2. Review error messages in `*.err` files
3. Verify all prerequisites are installed (`requirements.txt`)
