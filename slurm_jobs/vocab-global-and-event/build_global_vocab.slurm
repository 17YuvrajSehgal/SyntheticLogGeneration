#!/bin/bash
#SBATCH --job-name=build_vocab
#SBATCH --account=def-naser2
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/build_vocab_%j.out
#SBATCH --error=logs/build_vocab_%j.err

set -euo pipefail

# Modules
module purge
module load StdEnv/2023
module load python/3.11
module load arrow/21.0.0

# Paths
PROJECT_DIR="/project/def-naser2/yuvraj/SyntheticLogGeneration"
SCRIPT="$PROJECT_DIR/synthetic_data_processing/build_training_vocabs.py"
PARQUET_ROOT="$SCRATCH/enriched_parquet"
OUTPUT_DIR="$PROJECT_DIR/dataset/metadata_all_events"

echo "[INFO] Building Global Vocabulary from $PARQUET_ROOT..."
echo "[INFO] Output: $OUTPUT_DIR"

python -u "$SCRIPT" \
  --parquet-dir "$PARQUET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --top-k-ret 1024

echo "[INFO] Done."
