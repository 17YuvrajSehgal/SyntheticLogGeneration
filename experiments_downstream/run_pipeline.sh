#!/bin/bash
#
# Complete pipeline for downstream experiments
#
# Usage:
#   bash experiments_downstream/run_pipeline.sh <benchmark> <context_len>
#
# Example:
#   bash experiments_downstream/run_pipeline.sh scimark2 256
#   bash experiments_downstream/run_pipeline.sh pybench 1024
#

set -e  # Exit on error

# ============================================================
# Parse Arguments
# ============================================================
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <benchmark> <context_len>"
    echo "Example: $0 scimark2 256"
    exit 1
fi

BENCHMARK=$1
CONTEXT_LEN=$2

echo "============================================================"
echo "Downstream Experiments Pipeline"
echo "============================================================"
echo "Benchmark: $BENCHMARK"
echo "Context Length: $CONTEXT_LEN"
echo "============================================================"
echo ""

# ============================================================
# Configuration
# ============================================================
SCRATCH=${SCRATCH:-/scratch/yuvraj17}
BASE_DIR="$SCRATCH/SyntheticLogGeneration"
RESULTS_DIR="$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/$BENCHMARK/$CONTEXT_LEN"
DATA_DIR="$RESULTS_DIR/data"
RESULTS_OUT_DIR="$RESULTS_DIR/results"

# Paths
REAL_GLOB="/scratch/yuvraj17/windowed_npz_${CONTEXT_LEN}/${BENCHMARK}/train/*.npz"
CHECKPOINT="$BASE_DIR/experiments_results/${BENCHMARK}/exp_context_${CONTEXT_LEN}/exp_context_${CONTEXT_LEN}_${BENCHMARK}/exp_context_${CONTEXT_LEN}_${BENCHMARK}/ckpt_epoch_19.pt"
CONSTRAINTS="dataset/constraints_universal.json"

# Create output directories
mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_OUT_DIR"

echo "Data directory: $DATA_DIR"
echo "Results directory: $RESULTS_OUT_DIR"
echo ""

# ============================================================
# STEP 1: Prepare real train/test data
# ============================================================
echo "============================================================"
echo "STEP 1: Preparing Real Train/Test Data"
echo "============================================================"

python experiments_downstream/prepare_data.py \
    --real-glob "$REAL_GLOB" \
    --benchmark "$BENCHMARK" \
    --constraints "$CONSTRAINTS" \
    --output-dir "$DATA_DIR"

echo ""

# ============================================================
# STEP 2: Generate synthetic data
# ============================================================
echo "============================================================"
echo "STEP 2: Generating 10k Synthetic Data"
echo "============================================================"

# Determine batch size based on context length
if [ "$CONTEXT_LEN" -le 256 ]; then
    BATCH_SIZE=64
elif [ "$CONTEXT_LEN" -le 1024 ]; then
    BATCH_SIZE=32
elif [ "$CONTEXT_LEN" -le 2048 ]; then
    BATCH_SIZE=16
else
    BATCH_SIZE=8
fi

echo "Using batch size: $BATCH_SIZE for context length $CONTEXT_LEN"

python sample_diffusion.py \
    --ckpt "$CHECKPOINT" \
    --out "$DATA_DIR/synthetic_raw_${CONTEXT_LEN}_10k.npz" \
    --num-samples 10000 \
    --seq-len "$CONTEXT_LEN" \
    --batch-size "$BATCH_SIZE" \
    --use-ddim \
    --ddim-steps 50

echo ""

# ============================================================
# STEP 3: Repair synthetic data
# ============================================================
echo "============================================================"
echo "STEP 3: Repairing Synthetic Data"
echo "============================================================"

python synthetic_log_gen/repair.py \
    --trace "$DATA_DIR/synthetic_raw_${CONTEXT_LEN}_10k.npz" \
    --constraints "$CONSTRAINTS" \
    --output "$DATA_DIR/synthetic_repaired_${CONTEXT_LEN}_10k.npz"

echo ""

# ============================================================
# STEP 4: Create combined dataset
# ============================================================
echo "============================================================"
echo "STEP 4: Creating Combined Dataset (50/50)"
echo "============================================================"

python experiments_downstream/combine_datasets.py \
    --real-data "$DATA_DIR/real_train.npz" \
    --synthetic-data "$DATA_DIR/synthetic_repaired_${CONTEXT_LEN}_10k.npz" \
    --output "$DATA_DIR/combined_real_synthetic_${CONTEXT_LEN}_50_50.npz" \
    --ratio 0.5

echo ""

# ============================================================
# STEP 5: Train on real data (baseline)
# ============================================================
echo "============================================================"
echo "STEP 5: Training on Real Data (Baseline)"
echo "============================================================"

python experiments_downstream/models/train_predictor.py \
    --train-data "$DATA_DIR/real_train.npz" \
    --test-data "$DATA_DIR/real_test.npz" \
    --run-name "real_baseline_${BENCHMARK}_${CONTEXT_LEN}" \
    --output-dir "$RESULTS_OUT_DIR" \
    --seq-len 128 \
    --batch-size 64 \
    --epochs 20

echo ""

# ============================================================
# STEP 6: Train on synthetic only
# ============================================================
echo "============================================================"
echo "STEP 6: Training on Synthetic Only"
echo "============================================================"

python experiments_downstream/models/train_predictor.py \
    --train-data "$DATA_DIR/synthetic_repaired_${CONTEXT_LEN}_10k.npz" \
    --test-data "$DATA_DIR/real_test.npz" \
    --run-name "synthetic_only_${BENCHMARK}_${CONTEXT_LEN}" \
    --output-dir "$RESULTS_OUT_DIR" \
    --seq-len 128 \
    --batch-size 64 \
    --epochs 20 \
    --patience 10

echo ""

# ============================================================
# STEP 7: Train on combined data
# ============================================================
echo "============================================================"
echo "STEP 7: Training on Combined Data (50/50)"
echo "============================================================"

python experiments_downstream/models/train_predictor.py \
    --train-data "$DATA_DIR/combined_real_synthetic_${CONTEXT_LEN}_50_50.npz" \
    --test-data "$DATA_DIR/real_test.npz" \
    --run-name "combined_50_50_${BENCHMARK}_${CONTEXT_LEN}" \
    --output-dir "$RESULTS_OUT_DIR" \
    --seq-len 128 \
    --batch-size 64 \
    --epochs 20 \
    --patience 5

echo ""

# ============================================================
# DONE
# ============================================================
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Results saved to: $RESULTS_OUT_DIR"
echo ""
echo "Summary:"
echo "  - Real baseline: $RESULTS_OUT_DIR/real_baseline_${BENCHMARK}_${CONTEXT_LEN}/"
echo "  - Synthetic only: $RESULTS_OUT_DIR/synthetic_only_${BENCHMARK}_${CONTEXT_LEN}/"
echo "  - Combined: $RESULTS_OUT_DIR/combined_50_50_${BENCHMARK}_${CONTEXT_LEN}/"
echo ""
echo "To view results:"
echo "  cat $RESULTS_OUT_DIR/real_baseline_${BENCHMARK}_${CONTEXT_LEN}/final_metrics.json"
echo "  cat $RESULTS_OUT_DIR/synthetic_only_${BENCHMARK}_${CONTEXT_LEN}/final_metrics.json"
echo "  cat $RESULTS_OUT_DIR/combined_50_50_${BENCHMARK}_${CONTEXT_LEN}/final_metrics.json"
echo ""
