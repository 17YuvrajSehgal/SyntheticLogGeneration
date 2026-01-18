#!/bin/bash
#SBATCH --job-name=exp_ablation_system_pybench
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/%u/exp_ablation_system_pybench_4096_improved_%j.out
#SBATCH --error=/scratch/%u/exp_ablation_system_pybench_4096_improved_%j.err

set -euo pipefail

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "============================================================"
echo "Improved Ablation Model Training: pybench-4096"
echo "============================================================"
echo "JobID: $SLURM_JOB_ID"
echo "Node:  $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

# Activate virtual environment
cd $SCRATCH/SyntheticLogGeneration
source .venv/bin/activate

# Set environment variables
export SCRATCH=$SCRATCH
export PYTHONUNBUFFERED=1

# Training configuration
BENCHMARK="pybench"
CONTEXT_LEN=4096
DATA_ROOT="$SCRATCH/window_shards/windowed_npz_${CONTEXT_LEN}"
BATCH_SIZE=64
EPOCHS=20
RUN_NAME="improved_ablation_${BENCHMARK}_${CONTEXT_LEN}"

echo ""
echo "Configuration:"
echo "  Benchmark: $BENCHMARK"
echo "  Context Length: $CONTEXT_LEN"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Data Root: $DATA_ROOT"
echo "  Repetition Weight: 0.0 (BASELINE - no repetition loss)"
echo "  Transition Weight: 0.0 (BASELINE - no transition loss)"
echo "============================================================"
echo ""

# Train improved model with weights=0 (baseline comparison)
echo "Starting training..."
python train_experiment_better.py \
    --data-root "$DATA_ROOT" \
    --benchmark "$BENCHMARK" \
    --seq-len $CONTEXT_LEN \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --channels event dt cpu tid \
    --d-model 256 \
    --nhead 4 \
    --num-layers 4 \
    --repetition-weight 0.0 \
    --transition-weight 0.0 \
    --mixed-precision bf16 \
    --log-dir logs_tensorboard \
    --run-name "$RUN_NAME"

echo "[DONE] Finished exp_ablation_full_pybench_4096"