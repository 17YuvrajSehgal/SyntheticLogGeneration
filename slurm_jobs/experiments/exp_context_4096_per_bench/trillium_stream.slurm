#!/bin/bash
#SBATCH --job-name=stream_exp_context_4096
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=13:00:00
#SBATCH --output=/scratch/%u/slurm_%x_%j.out
#SBATCH --error=/scratch/%u/slurm_%x_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

# ============================================================
# TRILLIUM ASSUMPTION (same as your sample):
# Repo is cloned into:  $SCRATCH/SyntheticLogGeneration
# Venv exists at:       $SCRATCH/SyntheticLogGeneration/.venv
# ============================================================

REPO="$SCRATCH/SyntheticLogGeneration"
cd "$REPO"

# Activate scratch venv
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

BENCHMARK="stream"
RUN_NAME="exp_context_4096_stream"
CONTEXT_LEN="4096"

# Make a per-job run directory in scratch so outputs don't collide
RUN_DIR="$SCRATCH/SyntheticLogGeneration_runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Put outputs under the per-job run dir (Trillium style)
OUTPUT_DIR="$RUN_DIR/experiments_results"
EXPERIMENT_DIR="$OUTPUT_DIR/$BENCHMARK/exp_context_${CONTEXT_LEN}/$RUN_NAME"
mkdir -p "$EXPERIMENT_DIR"

echo "========================================"
echo "[START] Experiment: $RUN_NAME"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Repo: $REPO"
echo "Run Dir: $RUN_DIR"
echo "Benchmark: $BENCHMARK"
echo "Seq Len: $CONTEXT_LEN"
echo "Output Directory: $EXPERIMENT_DIR"
echo ""

# Input data expected on TRILLIUM scratch
DATA_ROOT="$SCRATCH/windowed_npz_${CONTEXT_LEN}"
REAL_GLOB="$DATA_ROOT/$BENCHMARK/train/*.npz"

# Fail fast if inputs are missing (same pattern as sample)
shopt -s nullglob
files=( $REAL_GLOB )
if [ ${#files[@]} -eq 0 ]; then
  echo "ERROR: No files matched:"
  echo "  $REAL_GLOB"
  echo "Did you copy windowed_npz_${CONTEXT_LEN} to Trillium \$SCRATCH?"
  exit 1
fi
shopt -u nullglob

python -u train_experiment.py \
    --data-root "$DATA_ROOT" \
    --benchmark "$BENCHMARK" \
    --seq-len 4096 \
    --channels event dt cpu tid comm ret \
    --model-type diffusion \
    --d-model 512 \
    --nhead 8 \
    --num-layers 8 \
    --batch-size 32 \
    --num-workers 8 \
    --num-cpus 4 \
    --tid-buckets 256 \
    --fd-cap 1025 \
    --epochs 20 \
    --steps 1000 \
    --lr 2e-4 \
    --max-steps-per-epoch 500 \
    --mixed-precision bf16 \
    --log-dir "$EXPERIMENT_DIR" \
    --run-name "$RUN_NAME"

echo ""
echo "[DONE] Finished at $(date)"
echo ""

# Persist results to HOME (scratch can be purged)
FINAL_OUT="$HOME/SyntheticLogGeneration_results_${SLURM_JOB_ID}"
mkdir -p "$FINAL_OUT"
rsync -a "$OUTPUT_DIR/" "$FINAL_OUT/"

echo "Copied results to:"
echo "  - $FINAL_OUT"
echo ""
echo "[DONE] Job completed successfully"
