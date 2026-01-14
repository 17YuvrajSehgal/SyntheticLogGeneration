# Downstream Task Experiments: Synthetic Data Utility Evaluation

This directory contains the complete experimental framework for evaluating the **utility of synthetic kernel traces** for downstream machine learning tasks.

## Research Questions

### RQ1: Data Utility
**Can synthetic traces replace real data for training downstream models?**

We compare F1 scores when training next-event prediction models on:
- Real data only (baseline)
- Synthetic data only (raw)
- Synthetic data only (repaired)
- Real + Synthetic (data augmentation)

**Expected Outcome**: Repaired synthetic data achieves >85% of baseline F1 score.

---

### RQ2: Repair Effectiveness
**Does constraint-guided repair improve downstream task performance?**

We measure the improvement: ΔF1 = F1(Repaired) - F1(Raw)

**Expected Outcome**: Repair improves F1 by >15 points.

---

### RQ3: Context Length Impact
**Does longer context improve downstream performance?**

We compare models trained on synthetic data from different context lengths:
- context_256 (short)
- context_1024 (medium)
- context_4096 (long)

**Expected Outcome**: Longer context produces higher-quality synthetic data.

---

### RQ4: Feature Ablation
**Which channels are most critical for utility?**

We compare:
- Event-only model
- Full feature model (event, dt, cpu, tid, comm, ret)

**Expected Outcome**: Full features outperform event-only by >15 points.

---

## Downstream Task: Next-Event Prediction

**Task**: Given a sequence of events `[e_1, ..., e_t]`, predict the next event `e_{t+1}`.

**Model**: Transformer-based classifier
- 4-layer Transformer encoder
- d_model=256, nhead=8
- Input: 128-event sequences
- Output: Softmax over 384 event classes

**Metrics**:
- **Primary**: F1-score (macro), F1-score (weighted), Accuracy
- **Secondary**: Top-5 accuracy, Top-10 accuracy
- **Diagnostic**: Per-class F1, confusion matrix

---

## Directory Structure

```
experiments_downstream/
├── README.md                   # This file
├── run_all_experiments.sh      # Master script to run everything
├── prepare_data.py             # Data preparation
├── analyze_results.py          # Results analysis
├── models/
│   ├── next_event_predictor.py # Model architecture
│   └── train_predictor.py      # Training script
├── data/                       # Generated data (gitignored)
│   ├── real_train.npz
│   ├── real_test.npz
│   ├── synthetic_raw_256.npz
│   ├── synthetic_repaired_256.npz
│   ├── synthetic_raw_1024.npz
│   ├── synthetic_repaired_1024.npz
│   ├── synthetic_raw_4096.npz
│   ├── synthetic_repaired_4096.npz
│   └── combined_real_synthetic_*.npz
├── results/                    # Training results (gitignored)
│   ├── real_baseline/
│   ├── synthetic_raw_1024/
│   ├── synthetic_repaired_1024/
│   ├── combined_1024/
│   └── ...
└── analysis/                   # Analysis outputs
    ├── summary_all_runs.csv
    ├── rq1_data_utility.csv
    ├── rq2_repair_effectiveness.csv
    ├── rq3_context_length.csv
    └── rq4_feature_ablation.csv
```

---

## Quick Start

### Prerequisites

1. **Trained diffusion models** for context lengths 256, 1024, 4096
   - Located in `experiments_results/exp_context_*/ckpt_epoch_99.pt`

2. **Real data** for the benchmark (e.g., scimark2)
   - Located in `dataset/window_shards/scimark2/train/*.npz`

3. **Learned constraints**
   - Located in `dataset/constraints_universal.json`

### Option 1: Run Everything (Recommended)

```bash
# Edit checkpoint paths in run_all_experiments.sh first!
bash experiments_downstream/run_all_experiments.sh
```

This will:
1. Prepare all data (split real, generate synthetic, repair)
2. Train all downstream models (~10-12 configurations)
3. Analyze results and generate tables

**Estimated Time**: 12-24 hours (depending on GPU)

---

### Option 2: Step-by-Step

#### Step 1: Prepare Data

**IMPORTANT**: Run this from the repository root (`SyntheticLogGeneration/`), not from `experiments_downstream/`

```bash
# Make sure you're in the repo root
cd c:/workplace/SyntheticLogGeneration

# Run data preparation
python experiments_downstream/prepare_data.py \
    --real-glob "dataset/window_shards/windowed_npz_1024/scimark2/train/*.npz" \
    --benchmark scimark2 \
    --generate-synthetic \
    --checkpoint-256 experiments_results/exp_context_256/ckpt_epoch_99.pt \
    --checkpoint-1024 experiments_results/exp_context_1024/ckpt_epoch_99.pt \
    --checkpoint-4096 experiments_results/exp_context_4096/ckpt_epoch_99.pt \
    --num-synthetic-samples 10000 \
    --constraints dataset/constraints_universal.json \
    --output-dir experiments_downstream/data
```

**Output**: 
- `experiments_downstream/data/real_train.npz`, `experiments_downstream/data/real_test.npz`
- `experiments_downstream/data/synthetic_raw_*.npz`, `experiments_downstream/data/synthetic_repaired_*.npz`
- `experiments_downstream/data/combined_real_synthetic_*.npz`

---

#### Step 2: Train Downstream Models

**IMPORTANT**: Run from repository root

**Baseline (Real data only)**:
```bash
python experiments_downstream/models/train_predictor.py \
    --train-data experiments_downstream/data/real_train.npz \
    --test-data experiments_downstream/data/real_test.npz \
    --run-name real_baseline \
    --output-dir experiments_downstream/results \
    --seq-len 128 --batch-size 64 --epochs 20
```

**Synthetic (Repaired)**:
```bash
python experiments_downstream/models/train_predictor.py \
    --train-data experiments_downstream/data/synthetic_repaired_1024.npz \
    --test-data experiments_downstream/data/real_test.npz \
    --run-name synthetic_repaired_1024 \
    --output-dir experiments_downstream/results \
    --seq-len 128 --batch-size 64 --epochs 20
```

**Combined (Real + Synthetic)**:
```bash
python experiments_downstream/models/train_predictor.py \
    --train-data experiments_downstream/data/combined_real_synthetic_1024.npz \
    --test-data experiments_downstream/data/real_test.npz \
    --run-name combined_1024 \
    --output-dir experiments_downstream/results \
    --seq-len 128 --batch-size 64 --epochs 20
```

Repeat for other configurations (raw synthetic, different context lengths, etc.)

---

#### Step 3: Analyze Results

```bash
python experiments_downstream/analyze_results.py \
    --results-dir experiments_downstream/results \
    --output-dir experiments_downstream/analysis
```

**Output**:
- `analysis/summary_all_runs.csv` - All results
- `analysis/rq1_data_utility.csv` - RQ1 table
- `analysis/rq2_repair_effectiveness.csv` - RQ2 table
- `analysis/rq3_context_length.csv` - RQ3 table
- `analysis/rq4_feature_ablation.csv` - RQ4 table

---

## Expected Results

Based on the experimental design, here are the expected F1 scores (macro):

| Configuration | Expected F1 | Purpose |
|---------------|-------------|---------|
| Real (Baseline) | 0.90-0.95 | Upper bound |
| Synthetic (Raw) | 0.60-0.75 | Without repair |
| Synthetic (Repaired) | 0.80-0.90 | With repair |
| Real + Synthetic | 0.92-0.96 | Data augmentation |

**Key Findings**:
1. ✓ Repaired synthetic achieves >85% of baseline F1
2. ✓ Repair improves F1 by >15 points
3. ✓ Longer context (4096) outperforms short context (256)
4. ✓ Full features outperform event-only

---

## Customization

### Use a Different Benchmark

Edit the `BENCHMARK` variable in `run_all_experiments.sh`:
```bash
BENCHMARK="compress-gzip"  # or ffmpeg, iozone, etc.
```

### Change Model Architecture

Edit `train_predictor.py` arguments:
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

### Issue: Synthetic data not found

**Solution**: Generate synthetic data first
```bash
python sample_diffusion.py \
    --ckpt experiments_results/exp_context_1024/ckpt_epoch_99.pt \
    --out experiments_downstream/data/synthetic_raw_1024.npz \
    --num-samples 10000 --seq-len 1024
```

### Issue: Training too slow

**Solution**: Use fewer epochs or enable mixed precision
```bash
--epochs 10 --mixed-precision bf16
```

---

## For the Paper

### Tables to Include

1. **Table 1 (RQ1)**: Data Utility Comparison
   - Source: `analysis/rq1_data_utility.csv`
   - Shows F1 scores for Real, Synthetic (Raw), Synthetic (Repaired), Combined

2. **Table 2 (RQ2)**: Repair Effectiveness
   - Source: `analysis/rq2_repair_effectiveness.csv`
   - Shows ΔF1 from repair for each context length

3. **Table 3 (RQ3)**: Context Length Impact
   - Source: `analysis/rq3_context_length.csv`
   - Shows F1 scores for context 256, 1024, 4096

4. **Table 4 (RQ4)**: Feature Ablation
   - Source: `analysis/rq4_feature_ablation.csv`
   - Shows F1 scores for event-only vs full features

### Key Claims

1. **Claim 1**: "Repaired synthetic traces achieve 85-90% of real data performance for next-event prediction"
   - Evidence: RQ1 results

2. **Claim 2**: "Constraint-guided repair improves downstream task F1 by 15-20 points"
   - Evidence: RQ2 results

3. **Claim 3**: "Combining real and synthetic data outperforms real-only training"
   - Evidence: RQ1 combined results

4. **Claim 4**: "Longer context windows produce higher-quality synthetic data"
   - Evidence: RQ3 results

---

## Citation

If you use this experimental framework, please cite:

```bibtex
@inproceedings{sehgal2026synthetic,
  title={Synthetic Kernel Trace Generation using Diffusion Models with Constraint-Guided Repair},
  author={Sehgal, Yuvraj and others},
  booktitle={Proceedings of the ACM SIGSOFT International Symposium on Foundations of Software Engineering (FSE)},
  year={2026}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
