#!/bin/bash
# Master script to run all downstream task experiments
# This script runs the complete experimental pipeline for RQ1-RQ4

set -e  # Exit on error

echo "========================================"
echo "Downstream Task Experiments - Master Script"
echo "========================================"

# Configuration
BENCHMARK="scimark2"
CONTEXT_LEN="1024"  # Use 1024 as the baseline context length for real data
REAL_DATA_GLOB="dataset/window_shards/windowed_npz_${CONTEXT_LEN}/${BENCHMARK}/train/*.npz"
CONSTRAINTS="dataset/constraints_universal.json"
OUTPUT_DIR="experiments_downstream"

# Checkpoints (update these paths!)
CKPT_256="experiments_results/exp_context_256/ckpt_epoch_99.pt"
CKPT_1024="experiments_results/exp_context_1024/ckpt_epoch_99.pt"
CKPT_4096="experiments_results/exp_context_4096/ckpt_epoch_99.pt"

echo ""
echo "Configuration:"
echo "  Benchmark: $BENCHMARK"
echo "  Real Data: $REAL_DATA_GLOB"
echo "  Constraints: $CONSTRAINTS"
echo "  Output: $OUTPUT_DIR"
echo ""

# ========================================
# PHASE 1: Data Preparation
# ========================================
echo "========================================"
echo "PHASE 1: Data Preparation"
echo "========================================"

python experiments_downstream/prepare_data.py \
    --real-glob "${REAL_DATA_GLOB}" \
    --benchmark "$BENCHMARK" \
    --generate-synthetic \
    --checkpoint-256 "$CKPT_256" \
    --checkpoint-1024 "$CKPT_1024" \
    --checkpoint-4096 "$CKPT_4096" \
    --num-synthetic-samples 10000 \
    --constraints "$CONSTRAINTS" \
    --output-dir "${OUTPUT_DIR}/data"

echo ""
echo "[Done] Data preparation complete"
echo ""

# ========================================
# PHASE 2: Train Downstream Models
# ========================================
echo "========================================"
echo "PHASE 2: Training Downstream Models"
echo "========================================"

# Common arguments
COMMON_ARGS="--seq-len 128 --batch-size 64 --epochs 20 --patience 3 --lr 1e-4"
DATA_DIR="${OUTPUT_DIR}/data"
RESULTS_DIR="${OUTPUT_DIR}/results"

# RQ1: Data Utility
echo ""
echo "--- RQ1: Data Utility ---"
echo ""

# 1. Real baseline
echo "[1/5] Training on Real data (baseline)..."
python experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/real_train.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "real_baseline" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# 2. Synthetic (raw) - 1024
echo "[2/5] Training on Synthetic (raw, 1024)..."
python experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/synthetic_raw_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "synthetic_raw_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# 3. Synthetic (repaired) - 1024
echo "[3/5] Training on Synthetic (repaired, 1024)..."
python experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "synthetic_repaired_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# 4. Combined (Real + Synthetic)
echo "[4/5] Training on Combined (Real + Synthetic)..."
python experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/combined_real_synthetic_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "combined_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# 5. Synthetic on Synthetic (consistency check)
echo "[5/5] Training on Synthetic, testing on Synthetic..."
python experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --test-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --run-name "synthetic_self_eval" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# RQ2: Repair Effectiveness (already covered above for 1024)
# Just need to add 256 and 4096

echo ""
echo "--- RQ2: Repair Effectiveness (Context 256, 4096) ---"
echo ""

# Context 256
if [ -f "${DATA_DIR}/synthetic_raw_256.npz" ]; then
    echo "[1/4] Training on Synthetic (raw, 256)..."
    python experiments_downstream/models/train_predictor.py \
        --train-data "${DATA_DIR}/synthetic_raw_256.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_raw_256" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
    
    echo "[2/4] Training on Synthetic (repaired, 256)..."
    python experiments_downstream/models/train_predictor.py \
        --train-data "${DATA_DIR}/synthetic_repaired_256.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_repaired_256" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
fi

# Context 4096
if [ -f "${DATA_DIR}/synthetic_raw_4096.npz" ]; then
    echo "[3/4] Training on Synthetic (raw, 4096)..."
    python experiments_downstream/models/train_predictor.py \
        --train-data "${DATA_DIR}/synthetic_raw_4096.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_raw_4096" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
    
    echo "[4/4] Training on Synthetic (repaired, 4096)..."
    python experiments_downstream/models/train_predictor.py \
        --train-data "${DATA_DIR}/synthetic_repaired_4096.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_repaired_4096" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
fi

# RQ3: Context Length (already covered above)

# RQ4: Feature Ablation
echo ""
echo "--- RQ4: Feature Ablation ---"
echo ""

# Event-only baseline
echo "[1/1] Training event-only model on Real data..."
python experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/real_train.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "event_only_baseline" \
    --model-type "event_only" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

echo ""
echo "[Done] All training complete"
echo ""

# ========================================
# PHASE 3: Analysis
# ========================================
echo "========================================"
echo "PHASE 3: Analysis"
echo "========================================"

python experiments_downstream/analyze_results.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "${OUTPUT_DIR}/analysis"

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Models: ${RESULTS_DIR}/"
echo "  - Analysis: ${OUTPUT_DIR}/analysis/"
echo ""
echo "Key files:"
echo "  - ${OUTPUT_DIR}/analysis/summary_all_runs.csv"
echo "  - ${OUTPUT_DIR}/analysis/rq1_data_utility.csv"
echo "  - ${OUTPUT_DIR}/analysis/rq2_repair_effectiveness.csv"
echo "  - ${OUTPUT_DIR}/analysis/rq3_context_length.csv"
echo "  - ${OUTPUT_DIR}/analysis/rq4_feature_ablation.csv"
echo ""
