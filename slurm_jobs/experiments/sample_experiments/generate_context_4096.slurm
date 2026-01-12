#!/bin/bash
#SBATCH --job-name=gen_ctx_4096
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/exp_context_4096/gen_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/exp_context_4096/gen_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"

echo "[START] Generating Samples for exp_context_4096"
date

# Checkpoint path (Using epoch 99 as verified)
CKPT="experiments_results/exp_context_4096/ckpt_epoch_99.pt"
OUT="generated_traces/context_4096_gen.npz"

# Create output dir if it doesn't exist (though script handles it too)
mkdir -p generated_traces

python -u sample_diffusion.py \
  --ckpt "$CKPT" \
  --out "$OUT" \
  --d-model 512 \
  --num-layers 8 \
  --seq-len 4096 \
  --num-samples 100 \
  --batch-size 32 \
  --num-cpus 4 \
  --tid-buckets 256 \
  --fd-cap 1025 \
  --device cuda

echo "[DONE] Saved to $OUT"
