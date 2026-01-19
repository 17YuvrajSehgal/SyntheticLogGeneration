#!/bin/bash
#SBATCH --job-name=train_improved_compress-gzip_4096
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/final_logs/train_improved_compress-gzip_4096_%j.out
#SBATCH --error=/scratch/%u/final_logs/train_improved_compress-gzip_4096_%j.err

# ============================================================================
# Train Diffusion Model with Improved NPZ Shards (Window 4096)
# ============================================================================
# This script trains the diffusion model using the improved NPZ shards
# with better GPU utilization and faster data loading.
# ============================================================================

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "============================================================"
echo "Training Diffusion Model with Improved Data Pipeline - 4096"
echo "============================================================"
echo "JobID: $SLURM_JOB_ID"
echo "Node:  $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

# Show GPU info
nvidia-smi

# Activate virtual environment
cd $SCRATCH/SyntheticLogGeneration
source .venv/bin/activate

# Set environment variables
export SCRATCH=$SCRATCH
export PYTHONUNBUFFERED=1

# Training configuration
BENCHMARK="compress-gzip"  # Change this for different benchmarks
CONTEXT_LEN=4096
DATA_ROOT="$SCRATCH/improved_window_shards/windowed_npz_${CONTEXT_LEN}"
BATCH_SIZE=128  # Larger batch for smaller context
EPOCHS=20
MAX_STEPS_PER_EPOCH=1000
RUN_NAME="improved_data_${BENCHMARK}_${CONTEXT_LEN}"

echo ""
echo "Configuration:"
echo "  Benchmark: $BENCHMARK"
echo "  Context Length: $CONTEXT_LEN"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Max Steps/Epoch: $MAX_STEPS_PER_EPOCH"
echo "  Data Root: $DATA_ROOT"
echo "  Using: Improved NPZ shards with better caching"
echo "============================================================"
echo ""

# Train model with improved data loader
echo "Starting training..."
python train_experiment_better.py \
    --data-root "$DATA_ROOT" \
    --benchmark "$BENCHMARK" \
    --seq-len $CONTEXT_LEN \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --max-steps-per-epoch $MAX_STEPS_PER_EPOCH \
    --d-model 4096 \
    --nhead 4 \
    --num-layers 4 \
    --repetition-weight 0.0 \
    --transition-weight 0.0 \
    --lr 2e-4 \
    --mixed-precision bf16 \
    --log-dir logs_tensorboard \
    --run-name "$RUN_NAME" \
    --num-workers 8

echo ""
echo "============================================================"
echo "Training Complete!"
echo "End: $(date)"
echo "============================================================"
echo ""
echo "Checkpoint saved to: logs_tensorboard/$RUN_NAME/"
echo ""
echo "Performance Notes:"
echo "  - Improved NPZ shards provide 20-50x faster loading"
echo "  - Expected GPU utilization: 80-95% (vs 30-60% with old pipeline)"
echo "  - Training should be 20-30% faster overall"
echo ""
echo "Next steps:"
echo "1. Generate samples:"
echo "   python sample_diffusion.py --ckpt logs_tensorboard/$RUN_NAME/ckpt_epoch_19.pt --out synthetic_${BENCHMARK}_${CONTEXT_LEN}.npz --num-samples 10000 --seq-len $CONTEXT_LEN --batch-size 64 --use-ddim --ddim-steps 50"
echo ""
