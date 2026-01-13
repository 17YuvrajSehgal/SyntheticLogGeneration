#!/bin/bash
#SBATCH --job-name=exp_ablation_full_iozone
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/exp_ablation_full_iozone/slurm_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/exp_ablation_full_iozone/slurm_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

# Ensure specific experiment dir exists
EXPERIMENT_DIR="$REPO/experiments_results/exp_ablation_full_iozone"
mkdir -p "$EXPERIMENT_DIR"

echo "[START] Experiment: exp_ablation_full_iozone"
echo "Date: $(date)"
echo "Output Directory: $EXPERIMENT_DIR"

# Note: train_experiment.py joins log-dir and run-name.
# So --log-dir experiments_results --run-name exp_ablation_full_iozone
# results in experiments_results/exp_ablation_full_iozone

python -u train_experiment.py \
    --data-root "$SCRATCH/windowed_npz_1024" \
    --benchmark "iozone" \
    --seq-len 1024 \
    --channels event dt cpu tid comm ret \
    --model-type diffusion \
    --d-model 512 \
    --nhead 8 \
    --num-layers 8 \
    --batch-size 256 \
    --num-cpus 4\
    --tid-buckets 256\
    --fd-cap 1025 \
    --epochs 100 \
    --steps 1000 \
    --lr 2e-4 \
    --mixed-precision bf16 \
    --log-dir experiments_results_iozone \
    --run-name exp_ablation_full_iozone

echo "[DONE] Finished exp_ablation_full_iozone"
