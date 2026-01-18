# Ablation Study Pipeline Documentation

## Overview

The **Ablation Study Pipeline** (`run_ablation_pipeline.py`) is an automated workflow that conducts a comprehensive cross-model evaluation to determine which input channels are most critical for downstream prediction tasks. This pipeline implements Cross-Model Evaluation Matrix, systematically testing all combinations of diffusion model channel configurations with predictor channel configurations.

## Purpose

This pipeline answers the research question: **"Which input channels are most critical for downstream prediction, and how does the diffusion model's channel configuration affect synthetic data quality?"**

It evaluates:
1. **Diffusion model channel importance**: Does modeling more channels produce better synthetic data?
2. **Predictor channel importance**: Which channels are most informative for prediction?
3. **Cross-model interactions**: Can a predictor trained on fewer channels still perform well with synthetic data from a full-channel diffusion model?

---

## Research Questions Addressed

- **RQ1**: Does modeling more channels in the diffusion model improve synthetic data quality?
- **RQ2**: Which channels (event, dt, cpu, tid, comm, ret) are most critical for downstream prediction?
- **RQ3**: Is there an optimal mismatch between diffusion model channels and predictor channels?
- **RQ4**: Do simpler models (fewer channels) provide sufficient utility with lower computational cost?

---

## Experimental Design

### Three Diffusion Model Variants

| Model | Channels Modeled | Description |
|-------|------------------|-------------|
| **Base** | event + dt (2 channels) | Minimal configuration - event type and timing |
| **System** | event + dt + cpu + tid (4 channels) | System-level context added |
| **Full** | event + dt + cpu + tid + comm + ret (6 channels) | Complete trace information |

### Four Predictor Configurations

| Configuration | Channels Used | Description |
|---------------|---------------|-------------|
| **event** | event only | Baseline - event sequence only |
| **event+dt** | event + dt | Timing-aware prediction |
| **event+dt+cpu+tid** | 4 channels | System context included |
| **all 6** | All 6 channels | Full information |

### Cross-Evaluation Matrix (9 Experiments)

|  | event | event+dt | event+dt+cpu+tid | all 6 |
|---|---|---|---|---|
| **Base** | ✅ | ✅ | ❌ | ❌ |
| **System** | ✅ | ✅ | ✅ | ❌ |
| **Full** | ✅ | ✅ | ✅ | ✅ |

**Note**: Predictors can only use channels that the diffusion model was trained to generate.

---

## Pipeline Architecture

### Execution Phases

#### **Step 0: Data Preparation** (Sequential)
Prepare real data for training and testing.

#### **Phase 1: Synthetic Generation** (Parallel - Steps 1-3)
Generate synthetic data from 3 diffusion models in parallel.

#### **Phase 2: Dataset Combination** (Sequential - Steps 4-6)
Create hybrid datasets (50% real + 50% synthetic) for each diffusion model.

#### **Phase 3: Cross-Evaluation Training** (Parallel - Steps 7-15)
Train 9 predictor configurations in parallel (up to 9 concurrent jobs).

---

## Pipeline Steps

### Step 0: Prepare Real Data
**Purpose**: Create train/test splits from real kernel traces

**Input**:
- Raw windowed traces: `$SCRATCH/window_shards/windowed_npz_1024/{benchmark}/train/*.npz`

**Output**:
- `real_train.npz` - Training set (80%)
- `real_test.npz` - Test set (20%)

**Channels**: All 6 channels (event, dt, cpu, tid, comm, ret)

**Note**: This step is shared across all experiments and only needs to run once per benchmark.

---

### Phase 1: Generate Synthetic Data (Parallel)

#### Step 1: Generate from Base Model
**Purpose**: Generate synthetic traces using base diffusion model (event + dt)

**Input**:
- Checkpoint: `logs_tensorboard/improved_ablation_{benchmark}_4096_base/ckpt_epoch_{N}.pt`

**Output**:
- `synthetic_base_10k.npz` (10,000 samples)

**Channels Generated**: event, dt (2 channels)

**Model Config**: d_model=256, nhead=4, num_layers=4, DDIM 50 steps

---

#### Step 2: Generate from System Model
**Purpose**: Generate synthetic traces using system diffusion model (event + dt + cpu + tid)

**Input**:
- Checkpoint: `logs_tensorboard/improved_ablation_{benchmark}_4096_system/ckpt_epoch_{N}.pt`

**Output**:
- `synthetic_system_10k.npz` (10,000 samples)

**Channels Generated**: event, dt, cpu, tid (4 channels)

---

#### Step 3: Generate from Full Model
**Purpose**: Generate synthetic traces using full diffusion model (all 6 channels)

**Input**:
- Checkpoint: `logs_tensorboard/improved_ablation_{benchmark}_4096_full/ckpt_epoch_{N}.pt`

**Output**:
- `synthetic_full_10k.npz` (10,000 samples)

**Channels Generated**: event, dt, cpu, tid, comm, ret (6 channels)

---

### Phase 2: Create Hybrid Datasets (Sequential)

#### Step 4: Create Hybrid (Base)
**Purpose**: Combine real data with base model synthetic data

**Input**:
- `real_train.npz`
- `synthetic_base_10k.npz`

**Output**:
- `hybrid_base_50_50.npz`

**Mixing Ratio**: 50% real + 50% synthetic

---

#### Step 5: Create Hybrid (System)
**Purpose**: Combine real data with system model synthetic data

**Input**:
- `real_train.npz`
- `synthetic_system_10k.npz`

**Output**:
- `hybrid_system_50_50.npz`

**Mixing Ratio**: 50% real + 50% synthetic

---

#### Step 6: Create Hybrid (Full)
**Purpose**: Combine real data with full model synthetic data

**Input**:
- `real_train.npz`
- `synthetic_full_10k.npz`

**Output**:
- `hybrid_full_50_50.npz`

**Mixing Ratio**: 50% real + 50% synthetic

---

### Phase 3: Cross-Evaluation Training (Parallel)

This phase trains 9 predictor configurations across the 3 hybrid datasets.

#### Row 1: Base Diffusion Model

**Step 7**: Base → event predictor
- **Training Data**: `hybrid_base_50_50.npz`
- **Predictor Channels**: event only
- **Run Name**: `cross_base_event`

**Step 8**: Base → event+dt predictor
- **Training Data**: `hybrid_base_50_50.npz`
- **Predictor Channels**: event, dt
- **Run Name**: `cross_base_base`

---

#### Row 2: System Diffusion Model

**Step 9**: System → event predictor
- **Training Data**: `hybrid_system_50_50.npz`
- **Predictor Channels**: event only
- **Run Name**: `cross_system_event`

**Step 10**: System → event+dt predictor
- **Training Data**: `hybrid_system_50_50.npz`
- **Predictor Channels**: event, dt
- **Run Name**: `cross_system_base`

**Step 11**: System → event+dt+cpu+tid predictor
- **Training Data**: `hybrid_system_50_50.npz`
- **Predictor Channels**: event, dt, cpu, tid
- **Run Name**: `cross_system_system`

---

#### Row 3: Full Diffusion Model

**Step 12**: Full → event predictor
- **Training Data**: `hybrid_full_50_50.npz`
- **Predictor Channels**: event only
- **Run Name**: `cross_full_event`

**Step 13**: Full → event+dt predictor
- **Training Data**: `hybrid_full_50_50.npz`
- **Predictor Channels**: event, dt
- **Run Name**: `cross_full_base`

**Step 14**: Full → event+dt+cpu+tid predictor
- **Training Data**: `hybrid_full_50_50.npz`
- **Predictor Channels**: event, dt, cpu, tid
- **Run Name**: `cross_full_system`

**Step 15**: Full → all 6 predictor
- **Training Data**: `hybrid_full_50_50.npz`
- **Predictor Channels**: event, dt, cpu, tid, comm, ret
- **Run Name**: `cross_full_full`

---

## Training Configuration

### Predictor Model
- **Architecture**: Transformer-based NextEventPredictor (or FlexibleNextEventPredictor for partial channels)
- **Task**: Next-event prediction (384 event classes)
- **Input Sequence Length**: 128 tokens
- **Batch Size**: 64
- **Epochs**: 20
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4

### Test Data
All experiments are evaluated on the same `real_test.npz` for fair comparison.

---

## Output Metrics

Each of the 9 experiments produces:

### Primary Metrics
- **F1 Score (Macro)**: Main metric - average F1 across all event classes
- **F1 Score (Weighted)**: Weighted by class frequency
- **Accuracy**: Overall prediction accuracy

### Secondary Metrics
- **Top-5 Accuracy**: Correct event in top 5 predictions
- **Top-10 Accuracy**: Correct event in top 10 predictions
- **Loss**: Cross-entropy loss

### Output Files (per experiment)
```
ablation-diffusion/{benchmark}/cross-results/{run_name}/
├── config.json           # Training configuration
├── final_metrics.json    # Final test metrics
├── history.json          # Training history
├── best_model.pt         # Best model checkpoint
└── predictions.npz       # Test set predictions
```

---

## Expected Results Pattern

### Scimark2 Example Results

| Diffusion ↓ / Predictor → | event | event+dt | event+dt+cpu+tid | all 6 |
|---------------------------|-------|----------|------------------|-------|
| **Base** | 67.19% | 66.46% | - | - |
| **System** | 66.81% | 66.35% | 68.57% | - |
| **Full** | 67.49% | 67.80% | **69.03%** | 67.14% |

### Key Findings

1. **Diagonal Performance**: Matching channels (Base→event+dt, System→4ch, Full→6ch) doesn't always yield best results
2. **Full Diffusion Wins**: Full diffusion model generally produces best synthetic data
3. **Optimal Predictor**: 4-channel predictor (event+dt+cpu+tid) often outperforms 6-channel
4. **Channel Paradox**: Adding comm+ret can hurt performance (noise vs signal)

---

## Usage

### Basic Usage
```bash
python run_ablation_pipeline.py --benchmark scimark2
```

### Advanced Usage
```bash
python run_ablation_pipeline.py \
    --benchmark ffmpeg \
    --base-epoch 19 \
    --system-epoch 18 \
    --full-epoch 19 \
    --num-samples 10000 \
    --max-parallel 9 \
    --skip-steps 1 2 3
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--benchmark` | Benchmark name (ffmpeg, pybench, scimark2) | Required |
| `--base-epoch` | Checkpoint epoch for base model | 19 |
| `--system-epoch` | Checkpoint epoch for system model | 19 |
| `--full-epoch` | Checkpoint epoch for full model | 19 |
| `--num-samples` | Number of synthetic samples per model | 10000 |
| `--max-parallel` | Max parallel training jobs | 9 |
| `--skip-steps` | Steps to skip (e.g., `1 2 3`) | None |
| `--scratch` | Scratch directory path | `$SCRATCH` |

---

## Results Analysis

### Collect Results
```bash
python collect_ablation_results.py --benchmark scimark2
```

### Output Files
```
ablation-diffusion/{benchmark}/cross-results/
├── ablation_summary_scimark2.txt       # Human-readable summary
├── ablation_detailed_scimark2.csv      # All metrics
├── ablation_matrix_scimark2.csv        # Cross-evaluation matrix
└── ablation_complete_scimark2.json     # Complete data
```

### Summary Report Includes
- Overall statistics (best/worst/mean F1)
- Cross-evaluation matrix (F1 percentages)
- Detailed results table
- Key findings:
  - Diagonal (matching) configurations
  - Best predictor per diffusion model
  - Best diffusion model per predictor

---

## Cluster Submission

### Generate Job Scripts
```bash
python generate_ablation_jobs.py
```

Creates Slurm scripts:
```
slurm_jobs/ablation/
├── run_ablation_ffmpeg.slurm
├── run_ablation_pybench.slurm
└── run_ablation_scimark2.slurm
```

### Submit Jobs
```bash
# Single benchmark
sbatch slurm_jobs/ablation/run_ablation_scimark2.slurm

# All benchmarks
for f in slurm_jobs/ablation/*.slurm; do sbatch $f; done
```

### Slurm Configuration
- **GPUs**: 1 per job
- **CPUs**: 16 (for parallel training)
- **Time**: 6 hours
- **Memory**: Auto-allocated

---

## Time Estimates

| Phase | Execution | Duration |
|-------|-----------|----------|
| Step 0 | Sequential | ~5 min |
| Phase 1 (3 generations) | Parallel | ~30 min |
| Phase 2 (3 combinations) | Sequential | ~5 min |
| Phase 3 (9 trainings) | Parallel | ~30 min |
| **Total** | | **~1.25 hours** |

**Speedup**: Parallel execution saves ~3.5 hours vs sequential (4.5 hrs → 1.25 hrs)

---

## Directory Structure

```
experiments_downstream_results/
├── ablation/{benchmark}/data/
│   ├── real_train.npz                    # Step 0
│   └── real_test.npz                     # Step 0
└── ablation-diffusion/{benchmark}/
    ├── synthetic_base_10k.npz            # Step 1
    ├── synthetic_system_10k.npz          # Step 2
    ├── synthetic_full_10k.npz            # Step 3
    ├── hybrid_base_50_50.npz             # Step 4
    ├── hybrid_system_50_50.npz           # Step 5
    ├── hybrid_full_50_50.npz             # Step 6
    └── cross-results/
        ├── cross_base_event/             # Step 7
        ├── cross_base_base/              # Step 8
        ├── cross_system_event/           # Step 9
        ├── cross_system_base/            # Step 10
        ├── cross_system_system/          # Step 11
        ├── cross_full_event/             # Step 12
        ├── cross_full_base/              # Step 13
        ├── cross_full_system/            # Step 14
        ├── cross_full_full/              # Step 15
        ├── ablation_summary_{benchmark}.txt
        ├── ablation_detailed_{benchmark}.csv
        ├── ablation_matrix_{benchmark}.csv
        └── ablation_complete_{benchmark}.json
```

---

## Troubleshooting

### Common Issues

**Issue**: Checkpoint not found for specific model
```
FileNotFoundError: .../improved_ablation_ffmpeg_4096_system/ckpt_epoch_19.pt
```
**Solution**: Specify different epoch for that model:
```bash
--base-epoch 19 --system-epoch 18 --full-epoch 19
```

**Issue**: Already generated synthetic data, want to skip
```bash
--skip-steps 1 2 3  # Skip Phase 1 (generation)
```

**Issue**: Want to run only specific training experiments
```bash
# Run only Full → all 6 (step 15)
--skip-steps 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
```

**Issue**: OOM during parallel training
```bash
--max-parallel 4  # Reduce concurrent jobs
```

---

## Interpretation Guide

### What to Look For

1. **Diagonal Trend**: Do matching configurations perform best?
   - Expected: Yes (optimal alignment)
   - Actual: Often no (simpler predictors can win)

2. **Row Comparison** (same predictor, different diffusion):
   - Does Full > System > Base?
   - Measures diffusion model quality

3. **Column Comparison** (same diffusion, different predictor):
   - Which channels are most informative?
   - Measures channel importance

4. **Best Overall**: Which configuration achieves highest F1?
   - Practical recommendation for deployment

### Example Insights (Scimark2)

- **Best**: Full → event+dt+cpu+tid (69.03%)
- **Worst**: System → event+dt (66.35%)
- **Range**: Only 2.68% - all configurations viable!
- **Finding**: comm+ret channels add noise, not signal

---

## Performance Optimization

### Parallel Execution
- **Phase 1**: 3 parallel jobs (generation)
- **Phase 3**: Up to 9 parallel jobs (training)
- **Speedup**: ~4.5x vs sequential

### Skip Completed Steps
```bash
# Already have synthetic data
--skip-steps 1 2 3

# Already have hybrids
--skip-steps 1 2 3 4 5 6

# Resume from specific step
--skip-steps 0 1 2 3 4 5 6 7 8 9 10  # Run steps 11-15 only
```

---

## Related Scripts

- **`run_pipeline.py`**: Main downstream evaluation pipeline
- **`collect_ablation_results.py`**: Aggregate and analyze results
- **`analyze_ablation_results.py`**: Legacy analysis script
- **`generate_ablation_jobs.py`**: Generate Slurm job scripts

---

## Citation

If you use this ablation study methodology in your research, please cite:

```bibtex
@inproceedings{sehgal2026ablation,
  title={Channel Importance in Synthetic Kernel Trace Generation},
  author={Sehgal, Yuvraj and others},
  booktitle={Proceedings of FSE},
  year={2026}
}
```

---

## Support

For issues or questions:
1. Check logs: `$SCRATCH/ablation_{benchmark}_{jobid}.out`
2. Review error files: `*.err`
3. Verify checkpoints exist in `logs_tensorboard/improved_ablation_*`
4. Ensure real data prepared: `experiments_downstream_results/ablation/{benchmark}/data/`
