#!/bin/bash
#SBATCH --job-name=sample_4096_scimark2
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/scimark2/exp_context_4096/sample_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/scimark2/exp_context_4096/sample_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"

echo "[START] Generating samples for scimark2 (context=4096)"
echo "Date: $(date)"

# Find the latest checkpoint (highest epoch number)
CKPT_DIR="experiments_results/scimark2/exp_context_4096/exp_context_4096_scimark2"
OUT="generated_traces/context_4096/scimark2_samples.npz"

# Create output directory
mkdir -p "generated_traces/context_4096"

# Check if checkpoint directory exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "[ERROR] Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

# Find all checkpoint files and extract epoch numbers
echo "[INFO] Searching for checkpoints in: $CKPT_DIR"
CKPT_FILES=("$CKPT_DIR"/ckpt_epoch_*.pt)

if [ ! -e "${CKPT_FILES[0]}" ]; then
    echo "[ERROR] No checkpoint files found in $CKPT_DIR"
    exit 1
fi

# Find checkpoint with highest epoch number
MAX_EPOCH=-1
BEST_CKPT=""

for ckpt_file in "${CKPT_FILES[@]}"; do
    # Extract epoch number from filename (e.g., ckpt_epoch_19.pt -> 19)
    filename=$(basename "$ckpt_file")
    epoch_num=$(echo "$filename" | sed -n 's/ckpt_epoch_\([0-9]*\)\.pt/\1/p')
    
    if [ -n "$epoch_num" ] && [ "$epoch_num" -gt "$MAX_EPOCH" ]; then
        MAX_EPOCH=$epoch_num
        BEST_CKPT="$ckpt_file"
    fi
done

if [ -z "$BEST_CKPT" ]; then
    echo "[ERROR] Could not find valid checkpoint files"
    exit 1
fi

CKPT="$BEST_CKPT"
echo "[INFO] Found ${#CKPT_FILES[@]} checkpoint(s)"
echo "[INFO] Using checkpoint with highest epoch: $CKPT (epoch $MAX_EPOCH)"
echo "[INFO] Output: $OUT"

python -u sample_diffusion.py \
    --ckpt "$CKPT" \
    --out "$OUT" \
    --d-model 512 \
    --nhead 8 \
    --num-layers 8 \
    --steps 1000 \
    --seq-len 4096 \
    --num-samples 1000 \
    --batch-size 32 \
    --vocab-dir dataset/metadata_all_events \
    --num-cpus 4 \
    --tid-buckets 256 \
    --fd-cap 1025 \
    --device cuda

echo "[DONE] Saved to $OUT"
echo "Date: $(date)"
