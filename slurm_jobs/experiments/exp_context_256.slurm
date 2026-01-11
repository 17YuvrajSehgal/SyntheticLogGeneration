#!/bin/bash
#SBATCH --job-name=exp_context_256
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=50:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/exp_context_256/slurm_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/exp_context_256/slurm_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

# Ensure specific experiment dir exists
EXPERIMENT_DIR="$REPO/experiments_results/exp_context_256"
mkdir -p "$EXPERIMENT_DIR"

echo "[START] Experiment: exp_context_256"
echo "Date: $(date)"
echo "Output Directory: $EXPERIMENT_DIR"

# Note: train_experiment.py joins log-dir and run-name.
# So --log-dir experiments_results --run-name exp_context_256 
# results in experiments_results/exp_context_256

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
    --log-dir experiments_results \
    --run-name exp_context_256

echo "[DONE] Finished exp_context_256"
