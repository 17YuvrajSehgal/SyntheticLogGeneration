#!/bin/bash
#SBATCH --job-name=train_diff_h100
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/diffusion_slurm/%x-%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/diffusion_slurm/%x-%j.err

set -euo pipefail

# 1. Environment Setup (Matched to reference)
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

# 2. Directory Creation
mkdir -p "$REPO/logs/diffusion_slurm"
mkdir -p "$REPO/logs_tensorboard"

echo "[START] Training Diffusion Model on H100"
echo "Date: $(date)"
echo "Node: $(hostname)"

# 3. Run Training
# Optimizations:
# - TF32 automatically enabled in script
# - BF16 mixed precision for H100
# - Compile enabled
# - Batch size 256 for high throughput

python -u train_experiment.py \
    --data-root "$SCRATCH/windowed_npz_256" \
    --benchmark "scimark2" \
    --seq-len 256 \
    --channels event dt cpu tid comm ret \
    --model-type diffusion \
    --d-model 512 \
    --nhead 8 \
    --num-layers 8 \
    --batch-size 256 \
    --epochs 100 \
    --steps 1000 \
    --lr 2e-4 \
    --mixed-precision bf16 \
    --compile \
    --log-dir logs_tensorboard \
    --run-name h100_diff_scimark2_256

echo "[DONE] Training finished."
echo "Date: $(date)"
