#!/bin/bash
#SBATCH --job-name=gen_npz
#SBATCH --account=def-naser2
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/gen_npz_%j.out
#SBATCH --error=logs/gen_npz_%j.err

set -euo pipefail

# Modules
module purge
module load StdEnv/2023
module load python/3.11
# PyArrow is needed for reading parquet
module load arrow/21.0.0

# Paths
PROJECT_DIR="/project/def-naser2/yuvraj/SyntheticLogGeneration"
SCRIPT="$PROJECT_DIR/synthetic_data_processing/parquet_to_windowed_npz.py"

# Inputs
INPUT_DIR="$SCRATCH/enriched_parquet"
VOCAB_DIR="$PROJECT_DIR/dataset/metadata_all_events"

# Output
OUTPUT_DIR="$SCRATCH/windowed_npz_1024"

# Parameters
SEQ_LEN=1024
STRIDE=512

echo "[INFO] Starting NPZ Generation..."
echo "       Input:  $INPUT_DIR"
echo "       Output: $OUTPUT_DIR"
echo "       Vocab:  $VOCAB_DIR"
echo "       Window: L=$SEQ_LEN, S=$STRIDE"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/logs"

python -u "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vocab-dir "$VOCAB_DIR" \
  --seq-len $SEQ_LEN \
  --stride $STRIDE

echo "[INFO] Done. Data is ready in $OUTPUT_DIR"
