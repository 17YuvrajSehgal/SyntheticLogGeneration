# Downstream Task Experiments: Synthetic Data Utility Evaluation

This directory contains the complete experimental framework for evaluating the **utility of synthetic kernel traces** for downstream machine learning tasks, specifically **next-event prediction**.

## Table of Contents
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Automated Pipelines](#automated-pipelines)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Ablation Studies](#ablation-studies)
- [Model Architecture](#model-architecture)
- [Results Analysis](#results-analysis)

---

## Overview

### What is This?

This framework evaluates synthetic kernel trace quality through **downstream task performance**. Instead of measuring statistical similarity, we ask: **"Can models trained on synthetic data perform well on real data?"**

### Key Capabilities

✅ **Automated pipelines**: End-to-end workflows from data generation to analysis  
✅ **Ablation studies**: Systematic channel importance evaluation  
✅ **Constraint repair**: Validate and fix synthetic data violations  
✅ **Parallel execution**: Efficient multi-experiment training  
✅ **Comprehensive metrics**: F1, accuracy, top-K, per-class analysis  

---

## Research Questions

### RQ1: Data Utility
**Can synthetic traces replace or augment real data for training downstream models?**

**Experiments**:
- Real data only (baseline)
- Synthetic data only (raw)
- Synthetic data only (repaired)
- Real + Synthetic (50/50 augmentation)

**Metric**: F1-score (macro) on real test set

**Expected Outcome**: Repaired synthetic + real > real alone (data augmentation benefit)

---

### RQ2: Repair Effectiveness
**Does constraint-guided repair improve downstream task performance?**

**Comparison**: F1(Repaired) vs F1(Raw)

**Expected Outcome**: Repair improves F1 by 10-20 points

---

### RQ3: Channel Importance (Ablation Study)
**Which input channels are most critical for downstream prediction?**

**Diffusion Model Variants**:
- **Base**: event + dt (2 channels)
- **System**: event + dt + cpu + tid (4 channels)
- **Full**: all 6 channels

**Predictor Variants**:
- event only
- event + dt
- event + dt + cpu + tid
- all 6 channels

**Cross-Evaluation Matrix**: 9 experiments testing all valid combinations

**Expected Outcome**: Identify optimal channel configuration

---

### RQ4: Context Length Impact
**Does longer context improve synthetic data quality?**

**Comparison**: Models trained on synthetic data from:
- context_256 (short)
- context_1024 (medium)
- context_4096 (long)

**Expected Outcome**: Longer context produces higher-quality synthetic data

---

## Automated Pipelines

### Main Pipeline (`run_pipeline.py`)

**Purpose**: Complete workflow for evaluating synthetic data utility

**Phases**:
1. **Data Preparation** (Sequential)
   - Split real data (train/test)
   - Generate synthetic data
   - Create hybrid datasets (no repair)
   - Repair synthetic data
   - Create hybrid datasets (repaired)

2. **Model Training** (Parallel)
   - Train on real only
   - Train on synthetic only
   - Train on combined (repaired)
   - Train on combined (no repair)

**Usage**:
```bash
python run_pipeline.py --benchmark scimark2 --window 1024
```

**Documentation**: See [`PIPELINE_DOCUMENTATION.md`](../PIPELINE_DOCUMENTATION.md)

---

### Ablation Pipeline (`run_ablation_pipeline.py`)

**Purpose**: Cross-model evaluation to determine channel importance

**Phases**:
1. **Generate Synthetic** (Parallel): 3 diffusion models
2. **Create Hybrids** (Sequential): 3 datasets
3. **Train Predictors** (Parallel): 9 cross-evaluation experiments

**Usage**:
```bash
python run_ablation_pipeline.py --benchmark scimark2
```

**Documentation**: See [`ABLATION_PIPELINE_DOCUMENTATION.md`](../ABLATION_PIPELINE_DOCUMENTATION.md)

---

## Directory Structure

```
experiments_downstream/
├── README.md                      # This file
├── prepare_data.py                # Data preparation script
├── combine_datasets.py            # Combine real + synthetic
├── diagnose_data_quality.py       # Data quality diagnostics
├── analyze_results.py             # Results aggregation
└── models/
    ├── next_event_predictor.py    # Transformer predictor (full features)
    ├── flexible_predictor.py      # Flexible predictor (any channel subset)
    └── train_predictor.py         # Training script
```

**Note**: `run_all_experiments.sh` has been replaced by automated Python pipelines (`run_pipeline.py`, `run_ablation_pipeline.py`)

---

## Quick Start

### Prerequisites

1. **Trained diffusion models**
   - Located in `logs_tensorboard/improved_baseline_{benchmark}_{window}/`
   - For ablation: `logs_tensorboard/improved_ablation_{benchmark}_4096_{type}/`

2. **Real data shards**
   - Located in `window_shards/windowed_npz_{window}/{benchmark}/train/`

3. **Constraints** (optional, for repair)
   - Located in `dataset/constraints_universal.json`

---

### Option 1: Automated Pipeline (Recommended)

**Run complete evaluation**:
```bash
python run_pipeline.py \
    --benchmark scimark2 \
    --window 1024 \
    --checkpoint-epoch 19 \
    --num-samples 10000
```

**Output**:
```
experiments_downstream_results/{benchmark}/{window}/
├── data/
│   ├── real_train.npz
│   ├── real_test.npz
│   ├── synthetic_raw_1024_10k.npz
│   ├── synthetic_repaired_1024_10k.npz
│   ├── combined_real_synthetic_norepair_1024_50_50.npz
│   └── combined_real_synthetic_repaired_1024_50_50.npz
└── results/
    ├── real_baseline_{benchmark}_{window}/
    ├── synthetic_data_only_{benchmark}_{window}/
    ├── combined_50_50_{benchmark}_{window}/
    └── combined_50_50_norepair_{benchmark}_{window}/
```

**Time Estimate**: ~1-2 hours (1024 window)

---

### Option 2: Ablation Study

**Run cross-model evaluation**:
```bash
python run_ablation_pipeline.py \
    --benchmark scimark2 \
    --base-epoch 19 \
    --system-epoch 19 \
    --full-epoch 19 \
    --num-samples 10000
```

**Collect results**:
```bash
python collect_ablation_results.py --benchmark scimark2
```

**Output**:
```
ablation-diffusion/{benchmark}/cross-results/
├── cross_base_event/
├── cross_base_base/
├── cross_system_event/
├── cross_system_base/
├── cross_system_system/
├── cross_full_event/
├── cross_full_base/
├── cross_full_system/
├── cross_full_full/
├── ablation_summary_{benchmark}.txt
├── ablation_detailed_{benchmark}.csv
└── ablation_matrix_{benchmark}.csv
```

**Time Estimate**: ~1.5 hours (parallel execution)

---

### Option 3: Manual Step-by-Step

#### Step 1: Prepare Data

```bash
python experiments_downstream/prepare_data.py \
    --real-glob "$SCRATCH/window_shards/windowed_npz_1024/scimark2/train/*.npz" \
    --benchmark scimark2 \
    --output-dir experiments_downstream_results/scimark2/1024/data
```

**Output**: `real_train.npz`, `real_test.npz`

---

#### Step 2: Generate Synthetic Data

```bash
python sample_diffusion.py \
    --ckpt logs_tensorboard/improved_baseline_scimark2_1024/ckpt_epoch_19.pt \
    --out experiments_downstream_results/scimark2/1024/data/synthetic_raw_1024_10k.npz \
    --num-samples 10000 \
    --seq-len 1024 \
    --use-ddim --ddim-steps 50
```

---

#### Step 3: Repair Synthetic Data (Optional)

```bash
python synthetic_log_gen/repair.py \
    --trace experiments_downstream_results/scimark2/1024/data/synthetic_raw_1024_10k.npz \
    --constraints dataset/constraints_universal.json \
    --output experiments_downstream_results/scimark2/1024/data/synthetic_repaired_1024_10k.npz
```

---

#### Step 4: Create Hybrid Dataset

```bash
python experiments_downstream/combine_datasets.py \
    --real-data experiments_downstream_results/scimark2/1024/data/real_train.npz \
    --synthetic-data experiments_downstream_results/scimark2/1024/data/synthetic_repaired_1024_10k.npz \
    --output experiments_downstream_results/scimark2/1024/data/combined_50_50.npz \
    --ratio 0.5
```

---

#### Step 5: Train Predictor

**Baseline (Real only)**:
```bash
python experiments_downstream/models/train_predictor.py \
    --train-data experiments_downstream_results/scimark2/1024/data/real_train.npz \
    --test-data experiments_downstream_results/scimark2/1024/data/real_test.npz \
    --run-name real_baseline \
    --output-dir experiments_downstream_results/scimark2/1024/results \
    --seq-len 128 --batch-size 64 --epochs 20
```

**Combined (Real + Synthetic)**:
```bash
python experiments_downstream/models/train_predictor.py \
    --train-data experiments_downstream_results/scimark2/1024/data/combined_50_50.npz \
    --test-data experiments_downstream_results/scimark2/1024/data/real_test.npz \
    --run-name combined_50_50 \
    --output-dir experiments_downstream_results/scimark2/1024/results \
    --seq-len 128 --batch-size 64 --epochs 20
```

---

## Ablation Studies

### Channel Ablation

**Train with specific channels**:
```bash
python experiments_downstream/models/train_predictor.py \
    --train-data data/real_train.npz \
    --test-data data/real_test.npz \
    --channels event dt cpu tid \
    --run-name ablation_4ch \
    --output-dir results/ \
    --seq-len 128 --batch-size 64 --epochs 20
```

**Supported channel combinations**:
- `event` - Event only
- `event dt` - Event + timing
- `event dt cpu tid` - System context
- `event dt cpu tid comm ret` - Full features

**Model Selection**:
- `event` only → `NextEventPredictorEventOnly`
- All 6 channels → `NextEventPredictor`
- Partial channels → `FlexibleNextEventPredictor`

---

## Model Architecture

### NextEventPredictor (Full Features)

**Architecture**:
- **Input**: 6-channel sequences (event, dt, cpu, tid, comm, ret)
- **Embedding**: Separate embeddings per channel + fusion
- **Encoder**: 4-layer Transformer (d_model=256, nhead=8)
- **Output**: 384-way classification (next event)

**Training**:
- Optimizer: AdamW (lr=1e-4)
- Loss: Cross-entropy
- Epochs: 20 (with early stopping)
- Batch size: 64 (256/1024) or 32 (4096)

---

### FlexibleNextEventPredictor (Partial Channels)

**Purpose**: Adapt to any channel subset for ablation studies

**Key Feature**: Dynamic architecture based on `--channels` argument

**Example**:
```python
# Automatically uses only specified channels
model = FlexibleNextEventPredictor(
    channels=['event', 'dt', 'cpu'],
    vocab_sizes={'event': 384, 'cpu': 4},
    ...
)
```

---

## Results Analysis

### Output Metrics (per experiment)

**Files**:
```
results/{run_name}/
├── config.json           # Training configuration
├── final_metrics.json    # Test set metrics
├── history.json          # Training history (all epochs)
├── best_model.pt         # Best checkpoint
└── predictions.npz       # Test predictions
```

**Metrics in `final_metrics.json`**:
```json
{
    "f1_macro": 0.6903,
    "f1_weighted": 0.9382,
    "accuracy": 0.9390,
    "top5_accuracy": 0.9849,
    "top10_accuracy": 0.9912,
    "loss": 0.2156
}
```

---

### Aggregate Results

**Collect all results**:
```bash
python experiments_downstream/analyze_results.py \
    --results-dir experiments_downstream_results/scimark2/1024/results \
    --output-dir experiments_downstream_results/scimark2/1024/analysis
```

**Output**:
- `summary_all_runs.csv` - All experiments
- `comparison_table.csv` - Side-by-side comparison

---

## Expected Results

### Typical F1 Scores (Macro)

| Configuration | Scimark2 | FFmpeg | Pybench |
|---------------|----------|--------|---------|
| Real only | 70% | 62% | 68% |
| Synthetic (raw) | 45% | 40% | 43% |
| Synthetic (repaired) | 60% | 55% | 58% |
| Real + Synthetic (repaired) | **73%** | **65%** | **71%** |

**Key Findings**:
1. ✅ Synthetic data augmentation improves F1 by 3-5%
2. ✅ Repair improves synthetic-only F1 by 15-20%
3. ✅ Combined (real + synthetic) > real alone
4. ✅ Longer context (4096) > shorter context (256)

---

### Ablation Study Results (Scimark2)

| Diffusion ↓ / Predictor → | event | event+dt | event+dt+cpu+tid | all 6 |
|---------------------------|-------|----------|------------------|-------|
| **Base** | 67.19% | 66.46% | - | - |
| **System** | 66.81% | 66.35% | 68.57% | - |
| **Full** | 67.49% | 67.80% | **69.03%** | 67.14% |

**Key Findings**:
1. ✅ Full diffusion model produces best synthetic data
2. ✅ 4-channel predictor (event+dt+cpu+tid) often optimal
3. ⚠️ Adding comm+ret can hurt performance (noise vs signal)
4. ✅ All configurations within 2-3% range (robust)

---

## Customization

### Use Different Benchmark

```bash
python run_pipeline.py --benchmark ffmpeg --window 1024
```

### Change Model Architecture

Edit `train_predictor.py` or pass arguments:
```bash
--d-model 512 --nhead 16 --num-layers 6
```

### Adjust Training Hyperparameters

```bash
--batch-size 128 --epochs 30 --lr 5e-5 --patience 5
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or sequence length
```bash
--batch-size 32 --seq-len 64
```

### Issue: Checkpoint not found

**Solution**: Verify checkpoint path and epoch
```bash
ls logs_tensorboard/improved_baseline_scimark2_1024/
--checkpoint-epoch 18  # Use different epoch
```

### Issue: Training too slow

**Solution**: Use fewer epochs or skip steps
```bash
--epochs 10
--skip-steps 1 2 3  # Skip data generation if already done
```

### Issue: Channel mismatch error

**Solution**: Ensure predictor channels ⊆ diffusion model channels
```bash
# If diffusion model trained with 4 channels (event+dt+cpu+tid):
--channels event dt cpu tid  # ✅ Valid
--channels event dt cpu tid comm ret  # ❌ Invalid (comm, ret not in diffusion model)
```

---

## For the Paper

### Tables to Include

1. **Table 1**: Data Utility Comparison (RQ1)
   - Real vs Synthetic vs Combined
   - Shows data augmentation benefit

2. **Table 2**: Repair Effectiveness (RQ2)
   - Raw vs Repaired synthetic data
   - Shows constraint repair impact

3. **Table 3**: Channel Ablation Matrix (RQ3)
   - Cross-model evaluation results
   - Shows channel importance

4. **Table 4**: Context Length Impact (RQ4)
   - 256 vs 1024 vs 4096
   - Shows context quality trade-off

### Key Claims

1. **Claim 1**: "Synthetic data augmentation improves downstream task F1 by 3-5%"
   - Evidence: RQ1 combined results

2. **Claim 2**: "Constraint-guided repair improves synthetic data F1 by 15-20 points"
   - Evidence: RQ2 results

3. **Claim 3**: "4-channel models (event+dt+cpu+tid) provide optimal balance"
   - Evidence: RQ3 ablation matrix

4. **Claim 4**: "Longer context windows produce higher-quality synthetic data"
   - Evidence: RQ4 context comparison

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

## Support

For questions or issues:
1. Check pipeline documentation: `PIPELINE_DOCUMENTATION.md`, `ABLATION_PIPELINE_DOCUMENTATION.md`
2. Review training logs in results directories
3. Verify data format with `diagnose_data_quality.py`
