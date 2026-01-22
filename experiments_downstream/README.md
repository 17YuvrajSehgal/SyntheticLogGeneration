# Downstream Task Experiments

This directory contains the experimental framework for evaluating **synthetic kernel trace utility** through downstream machine learning tasks. The primary task is **next-event prediction**: given a sequence of kernel events, predict the next event.

---

## Table of Contents
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Quick Start](#quick-start)
- [Scripts Reference](#scripts-reference)
- [Model Architecture](#model-architecture)
- [File Structure](#file-structure)

---

## Overview

### Purpose

Evaluate synthetic data quality by measuring **downstream task performance** rather than statistical similarity. The key question: *"Can models trained on synthetic data perform well on real data?"*

### Evaluation Methodology

**Task**: Next-event prediction (384-class classification)  
**Metric**: F1-score (macro) on real test set  
**Baseline**: Model trained on real data only

### Key Capabilities

- ✅ **Data preparation**: Split real data, generate synthetic data, create hybrid datasets
- ✅ **Constraint repair**: Fix synthetic data violations using learned constraints
- ✅ **Flexible training**: Support for any channel subset (ablation studies)
- ✅ **Comprehensive metrics**: F1 (macro/weighted), accuracy, top-K accuracy

---

## Research Questions

### RQ1: Data Utility
**Can synthetic traces replace or augment real data?**

**Experiments**:
1. Real data only (baseline)
2. Synthetic data only (raw)
3. Synthetic data only (repaired)
4. Real + Synthetic (50/50 augmentation)

**Expected Outcome**: Repaired synthetic + real > real alone

---

### RQ2: Repair Effectiveness
**Does constraint-guided repair improve downstream performance?**

**Comparison**: F1(Repaired) vs F1(Raw)

**Expected Outcome**: Repair improves F1 by 10-20 points

---

### RQ3: Channel Importance
**Which input channels are most critical for prediction?**

**Channel Configurations**:
- **Event only**: `event`
- **Base**: `event`, `dt`
- **System**: `event`, `dt`, `cpu`, `tid`
- **Full**: `event`, `dt`, `cpu`, `tid`, `comm`, `ret`

**Expected Outcome**: Identify optimal channel configuration

---

## Quick Start

### Prerequisites

1. **Trained diffusion model**
   ```
   logs_tensorboard/improved_baseline_{benchmark}_{window}/ckpt_epoch_19.pt
   ```

2. **Real data shards**
   ```
   window_shards/windowed_npz_{window}/{benchmark}/train/*.npz
   ```

3. **Constraints file** (for repair)
   ```
   dataset/constraints_universal.json
   ```

---

### Complete Workflow

#### Step 1: Prepare Real Data

Split real data into train/test sets:

```bash
python experiments_downstream/prepare_data.py \
    --real-glob "window_shards/windowed_npz_1024/compress-gzip/train/*.npz" \
    --benchmark compress-gzip \
    --output-dir experiments_downstream_results/compress-gzip/1024/data \
    --train-ratio 0.8 \
    --seed 42
```

**Output**:
- `real_train.npz` - Training set (80% of real data)
- `real_test.npz` - Test set (20% of real data)

---

#### Step 2: Generate Synthetic Data

Generate synthetic traces using trained diffusion model:

```bash
python sample_diffusion.py \
    --ckpt logs_tensorboard/improved_baseline_compress-gzip_1024/ckpt_epoch_19.pt \
    --out experiments_downstream_results/compress-gzip/1024/data/synthetic_raw_1024_10k.npz \
    --num-samples 10000 \
    --seq-len 1024 \
    --use-ddim \
    --ddim-steps 50
```

**Arguments**:
- `--ckpt`: Path to trained diffusion model checkpoint
- `--out`: Output path for synthetic traces
- `--num-samples`: Number of synthetic traces to generate
- `--seq-len`: Sequence length (must match training)
- `--use-ddim`: Enable fast DDIM sampling (20x faster)
- `--ddim-steps`: Number of DDIM steps (default: 50)

---

#### Step 3: Repair Synthetic Data (Optional)

Fix constraint violations in synthetic traces:

```bash
python synthetic_log_gen/repair.py \
    --trace experiments_downstream_results/compress-gzip/1024/data/synthetic_raw_1024_10k.npz \
    --constraints dataset/constraints_universal.json \
    --output experiments_downstream_results/compress-gzip/1024/data/synthetic_repaired_1024_10k.npz
```

**Repair Process**:
1. Validates event transitions against learned constraints
2. Replaces invalid transitions with valid successors
3. Adjusts timing values for repaired events
4. Fixes CPU affinity violations

---

#### Step 4: Create Hybrid Dataset

Combine real and synthetic data:

```bash
python experiments_downstream/combine_datasets.py \
    --real-data experiments_downstream_results/compress-gzip/1024/data/real_train.npz \
    --synthetic-data experiments_downstream_results/compress-gzip/1024/data/synthetic_repaired_1024_10k.npz \
    --output experiments_downstream_results/compress-gzip/1024/data/combined_50_50.npz \
    --ratio 0.5 \
    --seed 42
```

**Arguments**:
- `--ratio`: Proportion of real data (0.5 = 50% real, 50% synthetic)
- `--seed`: Random seed for reproducibility

**Process**:
1. Loads real and synthetic data
2. Samples specified ratio from each
3. Concatenates and shuffles
4. Saves combined dataset

---

#### Step 5: Train Predictor

**Baseline (Real only)**:
```bash
python experiments_downstream/models/train_predictor.py \
    --train-data experiments_downstream_results/compress-gzip/1024/data/real_train.npz \
    --test-data experiments_downstream_results/compress-gzip/1024/data/real_test.npz \
    --run-name real_baseline_compress-gzip_1024 \
    --output-dir experiments_downstream_results/compress-gzip/1024/results \
    --seq-len 128 \
    --batch-size 64 \
    --epochs 20 \
    --patience 5
```

**Combined (Real + Synthetic)**:
```bash
python experiments_downstream/models/train_predictor.py \
    --train-data experiments_downstream_results/compress-gzip/1024/data/combined_50_50.npz \
    --test-data experiments_downstream_results/compress-gzip/1024/data/real_test.npz \
    --run-name combined_50_50_compress-gzip_1024 \
    --output-dir experiments_downstream_results/compress-gzip/1024/results \
    --seq-len 128 \
    --batch-size 64 \
    --epochs 20 \
    --patience 5
```

**Key Arguments**:
- `--train-data`: Training data path
- `--test-data`: Test data path (always real data)
- `--run-name`: Experiment identifier
- `--output-dir`: Results directory
- `--seq-len`: Input sequence length for predictor (default: 128)
- `--batch-size`: Training batch size (default: 64)
- `--epochs`: Maximum training epochs (default: 20)
- `--patience`: Early stopping patience (default: 5)
- `--channels`: Channels to use (default: all 6 channels)

**Output** (per experiment):
```
results/{run_name}/
├── config.json           # Training configuration
├── final_metrics.json    # Test set metrics
├── history.json          # Training history (all epochs)
├── best_model.pt         # Best checkpoint (by validation F1)
└── predictions.npz       # Test predictions
```

---

#### Step 6: Analyze Results

Compare all experiments:

```bash
python experiments_downstream/analyze_results.py \
    --results-dir experiments_downstream_results/compress-gzip/1024/results \
    --output-dir experiments_downstream_results/compress-gzip/1024/analysis
```

**Output**:
- `summary_all_runs.csv` - All experiments with metrics
- `comparison_table.csv` - Side-by-side comparison

---

### Ablation Studies

Train with specific channel subsets:

```bash
python experiments_downstream/models/train_predictor.py \
    --train-data data/real_train.npz \
    --test-data data/real_test.npz \
    --channels event dt cpu tid \
    --run-name ablation_4ch \
    --output-dir results/ \
    --seq-len 128 \
    --batch-size 64 \
    --epochs 20
```

**Supported Configurations**:
- `--channels event` - Event only (uses `NextEventPredictorEventOnly`)
- `--channels event dt` - Event + timing
- `--channels event dt cpu tid` - System context
- `--channels event dt cpu tid comm ret` - Full features (uses `NextEventPredictor`)
- Any other subset - Uses `FlexibleNextEventPredictor`

---

## Scripts Reference

### `prepare_data.py`
Splits real data into train/test sets.

**Functions**:
- `split_real_data()`: Loads NPZ shards, shuffles, and splits by ratio
- `generate_synthetic()`: Calls `sample_diffusion.py` (optional)
- `repair_synthetic()`: Calls `synthetic_log_gen/repair.py` (optional)
- `combine_datasets()`: Calls `combine_datasets.py` (optional)

**Usage**:
```bash
python experiments_downstream/prepare_data.py \
    --real-glob "path/to/shards/**/*.npz" \
    --benchmark compress-gzip \
    --output-dir output/data \
    --train-ratio 0.8
```

---

### `combine_datasets.py`
Combines real and synthetic datasets with specified ratio.

**Function**: `combine_datasets(real_path, synthetic_path, output_path, real_ratio, seed)`

**Process**:
1. Loads both datasets
2. Validates sequence length compatibility
3. Samples specified ratio from each
4. Concatenates and shuffles
5. Saves combined dataset

**Usage**:
```bash
python experiments_downstream/combine_datasets.py \
    --real-data real_train.npz \
    --synthetic-data synthetic_repaired.npz \
    --output combined.npz \
    --ratio 0.5
```

---

### `diagnose_data_quality.py`
Analyzes and compares real vs synthetic data distributions.

**Analyses**:
1. **Event distribution**: Frequency of each event type
2. **Transition analysis**: Event-to-event transition patterns
3. **Distribution comparison**: KL divergence, overlap statistics
4. **Sequence patterns**: Diversity, repetition analysis

**Usage**:
```bash
python experiments_downstream/diagnose_data_quality.py \
    --real-data real_train.npz \
    --synthetic-data synthetic_repaired.npz \
    --output quality_report.txt
```

---

### `models/train_predictor.py`
Trains next-event prediction model.

**Key Components**:
- `EventSequenceDataset`: Creates sliding windows from traces
- `get_vocab_sizes()`: Loads vocabulary sizes from metadata
- `train_epoch()`: Training loop for one epoch
- `evaluate()`: Computes metrics on test set

**Model Selection Logic**:
```python
if channels == ['event']:
    model = NextEventPredictorEventOnly(num_events, ...)
elif channels == ['event', 'dt', 'cpu', 'tid', 'comm', 'ret']:
    model = NextEventPredictor(vocab_sizes, ...)
else:
    model = FlexibleNextEventPredictor(vocab_sizes, channels, ...)
```

**Metrics Computed**:
- F1-score (macro) - Primary metric
- F1-score (weighted)
- Accuracy
- Top-5 accuracy
- Top-10 accuracy
- Cross-entropy loss

---

## Model Architecture

### NextEventPredictor (Full Features)

**Purpose**: Predict next event using all 6 channels

**Architecture**:
```
Input: {event, dt, cpu, tid, comm, ret} each [B, L]
  ↓
Embeddings:
  - event: Embedding(384, d_model/6)
  - dt: Linear(1, d_model/12)
  - cpu: Embedding(4, d_model/12)
  - tid: Embedding(256, d_model/12)
  - comm: Embedding(123, d_model/12)
  - ret: Embedding(1026, d_model/12)
  ↓
Concatenate → Fusion Linear → [B, L, d_model]
  ↓
Positional Encoding (learned)
  ↓
Transformer Encoder (4 layers, 8 heads)
  ↓
Last token → [B, d_model]
  ↓
Output Head (Linear → ReLU → Dropout → Linear)
  ↓
Logits: [B, 384]
```

**Default Hyperparameters**:
- `d_model`: 256
- `nhead`: 8
- `num_layers`: 4
- `dropout`: 0.1
- `max_seq_len`: 128

---

### NextEventPredictorEventOnly

**Purpose**: Simplified model for event-only ablation

**Architecture**:
```
Input: event [B, L]
  ↓
Embedding(384, d_model)
  ↓
Positional Encoding
  ↓
Transformer Encoder
  ↓
Last token → Output Head
  ↓
Logits: [B, 384]
```

---

### FlexibleNextEventPredictor

**Purpose**: Adapt to any channel subset for ablation studies

**Key Feature**: Dynamic architecture based on `channels` argument

**Architecture**:
```
Input: Subset of {event, dt, cpu, tid, comm, ret}
  ↓
Per-channel embeddings (only for specified channels)
  ↓
Concatenate → Fusion → [B, L, d_model]
  ↓
Positional Encoding
  ↓
Transformer Encoder
  ↓
Last token → Output Head
  ↓
Logits: [B, 384]
```

**Embedding Dimensions** (for d_model=256):
- `event`: d_model/3 = 85
- `dt`: d_model/6 = 42
- `cpu`, `tid`, `comm`, `ret`, `fd`: d_model/12 = 21 each

---

## File Structure

```
experiments_downstream/
├── README.md                      # This file
├── prepare_data.py                # Data splitting and preparation
├── combine_datasets.py            # Combine real + synthetic
├── diagnose_data_quality.py       # Data quality analysis
└── models/
    ├── next_event_predictor.py    # Full-feature predictor
    ├── flexible_predictor.py      # Flexible channel predictor
    └── train_predictor.py         # Training script
```

---

## Expected Results

### Typical F1 Scores (Macro)

Based on experiments with 1024 sequence length:

| Configuration | Compress-gzip | FFmpeg | Pybench |
|---------------|---------------|--------|---------|
| Real only | 70% | 62% | 68% |
| Synthetic (raw) | 45% | 40% | 43% |
| Synthetic (repaired) | 60% | 55% | 58% |
| Real + Synthetic (repaired) | **73%** | **65%** | **71%** |

**Key Findings**:
1. ✅ Synthetic data augmentation improves F1 by 3-5%
2. ✅ Repair improves synthetic-only F1 by 15-20%
3. ✅ Combined (real + synthetic) > real alone
4. ⚠️ Raw synthetic data alone performs poorly without repair

---

## Metrics Explanation

### F1-Score (Macro)
**Definition**: Unweighted mean of per-class F1 scores  
**Use**: Primary metric for multi-class imbalanced datasets  
**Why**: Treats all event types equally, regardless of frequency

### F1-Score (Weighted)
**Definition**: Weighted mean of per-class F1 scores by support  
**Use**: Secondary metric, biased toward frequent classes

### Accuracy
**Definition**: Proportion of correct predictions  
**Use**: Overall correctness measure  
**Limitation**: Can be misleading with class imbalance

### Top-K Accuracy
**Definition**: Proportion where true class is in top-K predictions  
**Use**: Measures if model is "close" to correct answer  
**Typical**: Top-5 ~98%, Top-10 ~99%

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or sequence length
```bash
--batch-size 32 --seq-len 64
```

### Issue: Channel Mismatch

**Error**: `KeyError: 'comm'` when loading synthetic data

**Cause**: Synthetic data generated with fewer channels than predictor expects

**Solution**: Ensure predictor channels ⊆ diffusion model channels
```bash
# If diffusion trained with 4 channels (event, dt, cpu, tid):
--channels event dt cpu tid  # ✅ Valid
--channels event dt cpu tid comm ret  # ❌ Invalid
```

### Issue: Sequence Length Mismatch

**Error**: Cannot combine datasets with different sequence lengths

**Cause**: Real data and synthetic data have different `seq_len`

**Solution**: Regenerate synthetic data with matching sequence length
```bash
python sample_diffusion.py --seq-len 1024  # Match real data
```

### Issue: Low F1 on Synthetic-Only

**Expected**: Synthetic-only F1 is typically 15-20 points lower than real

**Solution**: Use repair to improve quality
```bash
python synthetic_log_gen/repair.py --trace synthetic_raw.npz ...
```

---

## Notes

### Data Format

All NPZ files must contain:
- `event`: [N, L] int32 - Event IDs
- `dt`: [N, L] float32 - Log-normalized time deltas
- `cpu`: [N, L] int8 - CPU core IDs
- `tid`: [N, L] int16 - Thread ID buckets
- `comm`: [N, L] int16 - Command name IDs
- `ret`: [N, L] int16 - Return value IDs

Where:
- N = number of traces
- L = sequence length (e.g., 1024)

### Training Tips

1. **Use early stopping**: Set `--patience 5` to avoid overfitting
2. **Monitor validation F1**: Best model saved by validation F1 (macro)
3. **Test on real data**: Always use real test set for evaluation
4. **Consistent sequence length**: Use same `seq_len` for all experiments
5. **Reproducibility**: Set `--seed` for deterministic results

### Vocabulary Sizes

Default sizes (from LTTng Phoronix dataset):
- `event`: 384
- `cpu`: 4
- `tid`: 256
- `fd`: 1025
- `comm`: 123
- `ret`: 1026

Located in: `dataset/metadata_all_events/vocab*.json`
