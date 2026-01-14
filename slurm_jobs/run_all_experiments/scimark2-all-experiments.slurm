#!/bin/bash
#SBATCH --job-name=downstream_scimark2
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_downstream/logs/slurm_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_downstream/logs/slurm_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

# Create logs directory
mkdir -p "$REPO/experiments_downstream/logs"

echo "========================================"
echo "[START] Downstream Task Experiments"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Configuration
BENCHMARK="scimark2"
CONTEXT_LEN="1024"
REAL_DATA_GLOB="$SCRATCH/windowed_npz_${CONTEXT_LEN}/${BENCHMARK}/train/*.npz"
CONSTRAINTS="dataset/constraints_universal.json"
OUTPUT_DIR="experiments_downstream"

# Checkpoints
CKPT_256="experiments_results/exp_context_256/ckpt_epoch_99.pt"
CKPT_1024="experiments_results/exp_context_1024/ckpt_epoch_99.pt"
CKPT_4096="experiments_results/exp_context_4096/ckpt_epoch_99.pt"

echo "Configuration:"
echo "  Benchmark: $BENCHMARK"
echo "  Real Data: $REAL_DATA_GLOB"
echo "  Constraints: $CONSTRAINTS"
echo "  Output: $OUTPUT_DIR"
echo "  Checkpoints:"
echo "    - 256: $CKPT_256"
echo "    - 1024: $CKPT_1024"
echo "    - 4096: $CKPT_4096"
echo ""

# ========================================
# PHASE 1: Data Preparation
# ========================================
echo "========================================"
echo "PHASE 1: Data Preparation"
echo "========================================"
echo "Start time: $(date)"

python -u experiments_downstream/prepare_data.py \
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
echo "[Done] Data preparation complete at $(date)"
echo ""

# ========================================
# PHASE 2: Train Downstream Models
# ========================================
echo "========================================"
echo "PHASE 2: Training Downstream Models"
echo "========================================"
echo "Start time: $(date)"

# Common arguments
COMMON_ARGS="--seq-len 128 --batch-size 64 --epochs 20 --patience 3 --lr 1e-4"
DATA_DIR="${OUTPUT_DIR}/data"
RESULTS_DIR="${OUTPUT_DIR}/results"

# RQ1: Data Utility
echo ""
echo "--- RQ1: Data Utility ---"
echo ""

# 1. Real baseline
echo "[1/12] Training on Real data (baseline)..."
python -u experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/real_train.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "real_baseline" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# 2. Synthetic (raw) - 1024
echo "[2/12] Training on Synthetic (raw, 1024)..."
python -u experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/synthetic_raw_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "synthetic_raw_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# 3. Synthetic (repaired) - 1024
echo "[3/12] Training on Synthetic (repaired, 1024)..."
python -u experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "synthetic_repaired_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# 4. Combined (Real + Synthetic)
echo "[4/12] Training on Combined (Real + Synthetic)..."
python -u experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/combined_real_synthetic_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "combined_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# 5. Synthetic on Synthetic (consistency check)
echo "[5/12] Training on Synthetic, testing on Synthetic..."
python -u experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --test-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --run-name "synthetic_self_eval" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

# RQ2 & RQ3: Repair Effectiveness + Context Length
echo ""
echo "--- RQ2 & RQ3: Repair Effectiveness + Context Length ---"
echo ""

# Context 256
if [ -f "${DATA_DIR}/synthetic_raw_256.npz" ]; then
    echo "[6/12] Training on Synthetic (raw, 256)..."
    python -u experiments_downstream/models/train_predictor.py \
        --train-data "${DATA_DIR}/synthetic_raw_256.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_raw_256" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
    
    echo "[7/12] Training on Synthetic (repaired, 256)..."
    python -u experiments_downstream/models/train_predictor.py \
        --train-data "${DATA_DIR}/synthetic_repaired_256.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_repaired_256" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
fi

# Context 4096
if [ -f "${DATA_DIR}/synthetic_raw_4096.npz" ]; then
    echo "[8/12] Training on Synthetic (raw, 4096)..."
    python -u experiments_downstream/models/train_predictor.py \
        --train-data "${DATA_DIR}/synthetic_raw_4096.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_raw_4096" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
    
    echo "[9/12] Training on Synthetic (repaired, 4096)..."
    python -u experiments_downstream/models/train_predictor.py \
        --train-data "${DATA_DIR}/synthetic_repaired_4096.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_repaired_4096" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
fi

# RQ4: Feature Ablation
echo ""
echo "--- RQ4: Feature Ablation ---"
echo ""

# Event-only baseline
echo "[10/12] Training event-only model on Real data..."
python -u experiments_downstream/models/train_predictor.py \
    --train-data "${DATA_DIR}/real_train.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "event_only_baseline" \
    --model-type "event_only" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

echo ""
echo "[Done] All training complete at $(date)"
echo ""

# ========================================
# PHASE 3: Analysis
# ========================================
echo "========================================"
echo "PHASE 3: Analysis"
echo "========================================"
echo "Start time: $(date)"

python -u experiments_downstream/analyze_results.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "${OUTPUT_DIR}/analysis"

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo "End time: $(date)"
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
echo "[DONE] Job completed successfully"
