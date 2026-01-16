#!/bin/bash
#SBATCH --job-name=improved_scimark2_4096
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/improved_scimark2_4096_%j.out
#SBATCH --error=/scratch/%u/improved_scimark2_4096_%j.err

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "============================================================"
echo "Improved Diffusion Model Training: scimark2-4096"
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
BENCHMARK="scimark2"
CONTEXT_LEN=4096
DATA_ROOT="$SCRATCH/window_shards/windowed_npz_${CONTEXT_LEN}"
BATCH_SIZE=64
EPOCHS=100
RUN_NAME="improved_${BENCHMARK}_${CONTEXT_LEN}"

echo ""
echo "Configuration:"
echo "  Benchmark: $BENCHMARK"
echo "  Context Length: $CONTEXT_LEN"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Data Root: $DATA_ROOT"
echo "============================================================"
echo ""

# Train improved model
echo "Starting training..."
python train_experiment_better.py \
    --data-root "$DATA_ROOT" \
    --benchmark "$BENCHMARK" \
    --seq-len $CONTEXT_LEN \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --d-model 4096 \
    --nhead 4 \
    --num-layers 4 \
    --repetition-weight 0.05 \
    --transition-weight 0.03 \
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
echo "   python sample_diffusion.py --ckpt logs_tensorboard/$RUN_NAME/ckpt_epoch_99.pt --out synthetic_improved_4096.npz --num-samples 10000 --seq-len 4096 --batch-size 64 --use-ddim --ddim-steps 50"
echo ""
echo "2. Evaluate downstream:"
echo "   python experiments_downstream/models/train_predictor.py --train-data synthetic_improved_4096.npz --test-data real_test.npz --run-name synthetic_improved_4096"
echo ""
