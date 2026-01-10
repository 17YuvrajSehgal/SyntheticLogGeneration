#!/bin/bash
#SBATCH --job-name=extract_schemas
#SBATCH --account=def-naser2
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/extract_schemas/extract_schemas_%j.out
#SBATCH --error=logs/extract_schemas/extract_schemas_%j.err

set -euo pipefail

# Define paths
PROJECT_DIR="/project/def-naser2/yuvraj/SyntheticLogGeneration"
SCRIPT="$PROJECT_DIR/synthetic_data_processing/extract_event_schemas.py"

# Input is the scratch directory with all benchmarks
ROOT_DIR="$SCRATCH/txt_traces_all_benchmarks"

# Output goes to metadata_all_events
OUT_FILE="$PROJECT_DIR/dataset/metadata_all_events/event_schemas.json"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$(dirname "$OUT_FILE")"

cd "$PROJECT_DIR"

echo "[INFO] Starting schema extraction..."
echo "[INFO] Input Root: $ROOT_DIR"
echo "[INFO] Output: $OUT_FILE"

# Run the python script
python -u "$SCRIPT" \
  --dir "$ROOT_DIR" \
  --out "$OUT_FILE"

echo "[INFO] Job complete."
