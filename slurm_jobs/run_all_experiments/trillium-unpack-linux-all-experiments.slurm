#!/bin/bash
#SBATCH --job-name=downstream_unpack-linux
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/slurm_%x_%j.out
#SBATCH --error=/scratch/%u/slurm_%x_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

# ============================================================
# Run EVERYTHING from scratch
# Assumes you have already cloned the repo into:
#   $SCRATCH/SyntheticLogGeneration
# and created a venv there:
#   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# ============================================================

REPO="$SCRATCH/SyntheticLogGeneration"
cd "$REPO"

# Activate scratch venv
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

# Make a per-job run directory in scratch so outputs don't collide
RUN_DIR="$SCRATCH/SyntheticLogGeneration_runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Option 1 (recommended): run code in-place, write outputs into RUN_DIR
# (If you want a totally isolated copy per job, tell me and I'll switch to rsync.)
OUTPUT_DIR="$RUN_DIR/experiments_downstream"
DATA_DIR="$OUTPUT_DIR/data"
RESULTS_DIR="$OUTPUT_DIR/results"
ANALYSIS_DIR="$OUTPUT_DIR/analysis"
mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$ANALYSIS_DIR"

echo "========================================"
echo "[START] Downstream Task Experiments"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Repo: $REPO"
echo "Run Dir: $RUN_DIR"
echo ""

# Configuration
BENCHMARK="unpack-linux"
CONTEXT_LEN="1024"

# Input data expected on TRILLIUM scratch
REAL_DATA_GLOB="$SCRATCH/windowed_npz_${CONTEXT_LEN}/${BENCHMARK}/train/*.npz"

# These paths are relative to $REPO
CONSTRAINTS="$REPO/dataset/constraints_universal.json"
CKPT_256="$REPO/experiments_results/$BENCHMARK/exp_context_256/exp_context_256_unpack-linux/exp_context_256_unpack-linux/ckpt_epoch_19.pt"
CKPT_1024="$REPO/experiments_results/$BENCHMARK/exp_context_1024/exp_context_1024_unpack-linux/exp_context_1024_unpack-linux/ckpt_epoch_19.pt"
CKPT_4096="$REPO/experiments_results/$BENCHMARK/exp_context_4096/exp_context_4096_unpack-linux/exp_context_4096_unpack-linux/ckpt_epoch_19.pt"

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

# Fail fast if inputs are missing
shopt -s nullglob
files=( $REAL_DATA_GLOB )
if [ ${#files[@]} -eq 0 ]; then
  echo "ERROR: No files matched REAL_DATA_GLOB:"
  echo "  $REAL_DATA_GLOB"
  echo "Did you copy the windowed_npz_${CONTEXT_LEN} dataset to Trillium \$SCRATCH?"
  exit 1
fi
shopt -u nullglob

for f in "$CONSTRAINTS" "$CKPT_256" "$CKPT_1024" "$CKPT_4096"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Missing required file: $f"
    exit 1
  fi
done

# ========================================
# PHASE 1: Data Preparation
# ========================================
echo "========================================"
echo "PHASE 1: Data Preparation"
echo "========================================"
echo "Start time: $(date)"

python -u "$REPO/experiments_downstream/prepare_data.py" \
    --real-glob "${REAL_DATA_GLOB}" \
    --benchmark "$BENCHMARK" \
    --generate-synthetic \
    --checkpoint-256 "$CKPT_256" \
    --checkpoint-1024 "$CKPT_1024" \
    --checkpoint-4096 "$CKPT_4096" \
    --num-synthetic-samples 10000 \
    --constraints "$CONSTRAINTS" \
    --output-dir "$DATA_DIR"

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

COMMON_ARGS="--seq-len 128 --batch-size 64 --epochs 20 --patience 3 --lr 1e-4"

echo ""
echo "--- RQ1: Data Utility ---"
echo ""

echo "[1/12] Training on Real data (baseline)..."
python -u "$REPO/experiments_downstream/models/train_predictor.py" \
    --train-data "${DATA_DIR}/real_train.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "real_baseline" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

echo "[2/12] Training on Synthetic (raw, 1024)..."
python -u "$REPO/experiments_downstream/models/train_predictor.py" \
    --train-data "${DATA_DIR}/synthetic_raw_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "synthetic_raw_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

echo "[3/12] Training on Synthetic (repaired, 1024)..."
python -u "$REPO/experiments_downstream/models/train_predictor.py" \
    --train-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "synthetic_repaired_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

echo "[4/12] Training on Combined (Real + Synthetic)..."
python -u "$REPO/experiments_downstream/models/train_predictor.py" \
    --train-data "${DATA_DIR}/combined_real_synthetic_1024.npz" \
    --test-data "${DATA_DIR}/real_test.npz" \
    --run-name "combined_1024" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

echo "[5/12] Training on Synthetic, testing on Synthetic..."
python -u "$REPO/experiments_downstream/models/train_predictor.py" \
    --train-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --test-data "${DATA_DIR}/synthetic_repaired_1024.npz" \
    --run-name "synthetic_self_eval" \
    --output-dir "$RESULTS_DIR" \
    $COMMON_ARGS

echo ""
echo "--- RQ2 & RQ3: Repair Effectiveness + Context Length ---"
echo ""

if [ -f "${DATA_DIR}/synthetic_raw_256.npz" ]; then
    echo "[6/12] Training on Synthetic (raw, 256)..."
    python -u "$REPO/experiments_downstream/models/train_predictor.py" \
        --train-data "${DATA_DIR}/synthetic_raw_256.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_raw_256" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS

    echo "[7/12] Training on Synthetic (repaired, 256)..."
    python -u "$REPO/experiments_downstream/models/train_predictor.py" \
        --train-data "${DATA_DIR}/synthetic_repaired_256.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_repaired_256" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
fi

if [ -f "${DATA_DIR}/synthetic_raw_4096.npz" ]; then
    echo "[8/12] Training on Synthetic (raw, 4096)..."
    python -u "$REPO/experiments_downstream/models/train_predictor.py" \
        --train-data "${DATA_DIR}/synthetic_raw_4096.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_raw_4096" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS

    echo "[9/12] Training on Synthetic (repaired, 4096)..."
    python -u "$REPO/experiments_downstream/models/train_predictor.py" \
        --train-data "${DATA_DIR}/synthetic_repaired_4096.npz" \
        --test-data "${DATA_DIR}/real_test.npz" \
        --run-name "synthetic_repaired_4096" \
        --output-dir "$RESULTS_DIR" \
        $COMMON_ARGS
fi

echo ""
echo "--- RQ4: Feature Ablation ---"
echo ""

echo "[10/12] Training event-only model on Real data..."
python -u "$REPO/experiments_downstream/models/train_predictor.py" \
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

python -u "$REPO/experiments_downstream/analyze_results.py" \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$ANALYSIS_DIR"

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - Models:   ${RESULTS_DIR}/"
echo "  - Analysis: ${ANALYSIS_DIR}/"
echo ""

# Persist results to HOME (scratch can be purged)
FINAL_OUT="$HOME/SyntheticLogGeneration_results_${SLURM_JOB_ID}"
mkdir -p "$FINAL_OUT"
rsync -a "$OUTPUT_DIR/" "$FINAL_OUT/"

echo "Copied results to:"
echo "  - $FINAL_OUT"
echo ""
echo "[DONE] Job completed successfully"