#!/bin/bash
#SBATCH --job-name=gen_npz_4096
#SBATCH --account=def-naser2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/gen_npz_4096_%j.out
#SBATCH --error=logs/gen_npz_4096_%j.err

set -euo pipefail

# Modules
module purge
module load StdEnv/2023
module load python/3.11
module load arrow/21.0.0

# Paths
PROJECT_DIR="/project/def-naser2/yuvraj/SyntheticLogGeneration"
SCRIPT="$PROJECT_DIR/synthetic_data_processing/parquet_to_windowed_npz.py"
INPUT_DIR="$SCRATCH/enriched_parquet"
VOCAB_DIR="$PROJECT_DIR/dataset/metadata_all_events"
OUTPUT_DIR="$SCRATCH/windowed_npz_4096"

# Parameters
SEQ_LEN=4096
STRIDE=2048

echo "[INFO] Starting NPZ Generation (L=$SEQ_LEN, S=$STRIDE)..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/logs"

# Ensure we have enough memory for large windows
python -u "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vocab-dir "$VOCAB_DIR" \
  --seq-len $SEQ_LEN \
  --stride $STRIDE

echo "[INFO] Done."
