#!/bin/bash
#SBATCH --job-name=improved_baseline_phpbench_256
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/phpbench/exp_context_256/exp_context_256_phpbench/slurm_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/phpbench/exp_context_256/exp_context_256_phpbench/slurm_%j.err

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "============================================================"
echo "Improved Baseline Model Training: phpbench-256"
echo "============================================================"
echo "JobID: $SLURM_JOB_ID"
echo "Node:  $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

BENCHMARK="phpbench"
RUN_NAME="exp_context_256_phpbench"

# Put everything (including logs) inside the run folder
EXPERIMENT_DIR="$REPO/experiments_results/$BENCHMARK/exp_context_256/$RUN_NAME"
mkdir -p "$EXPERIMENT_DIR"

echo "[START] Experiment: $RUN_NAME"
echo "Benchmark: $BENCHMARK"
echo "Date: $(date)"
echo "Output Directory: $EXPERIMENT_DIR"

# Training configuration
BENCHMARK="phpbench"
CONTEXT_LEN=256
DATA_ROOT="$SCRATCH/window_shards/windowed_npz_${CONTEXT_LEN}"
BATCH_SIZE=64
EPOCHS=20
RUN_NAME="improved_baseline_${BENCHMARK}_${CONTEXT_LEN}"

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
python -u train_experiment_better.py \
    --data-root "$DATA_ROOT" \
    --benchmark "$BENCHMARK" \
    --seq-len $CONTEXT_LEN \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --d-model 256 \
    --nhead 4 \
    --num-layers 4 \
    --repetition-weight 0.0 \
    --transition-weight 0.0 \
    --mixed-precision bf16 \
    --log-dir logs_tensorboard \
    --run-name "$RUN_NAME"

echo ""
echo "============================================================"
echo "Training Complete!"
echo "End: $(date)"
echo "============================================================"
echo ""
echo "Checkpoint saved to: logs_tensorboard/$RUN_NAME/"
echo ""
echo "Next steps:"
echo "1. Generate samples:"
echo "   python sample_diffusion.py --ckpt logs_tensorboard/$RUN_NAME/ckpt_epoch_99.pt --out synthetic_baseline_256.npz --num-samples 10000 --seq-len 256 --batch-size 64 --use-ddim --ddim-steps 50"
echo ""
echo "2. Evaluate downstream:"
echo "   python experiments_downstream/models/train_predictor.py --train-data synthetic_baseline_256.npz --test-data real_test.npz --run-name synthetic_baseline_256"
echo ""
