#!/bin/bash
#SBATCH --job-name=improved_npz_4096
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/final_logs/improved_npz_4096_%j.out
#SBATCH --error=/scratch/%u/final_logs/improved_npz_4096_%j.err

set -euo pipefail

# ============================================================================
# Improved NPZ Generation (Window Size: 4096)
# ============================================================================
# This script converts parquet files to smaller NPZ shards for better GPU
# utilization during training. Each shard contains 500 windows (vs all windows
# in one file with the old pipeline).
#
# Expected improvements:
# - 20-50x faster data loading
# - 80-95% GPU utilization (vs 30-60% with spikes)
# - 20-30% faster training overall
# ============================================================================

echo "========================================="
echo "Improved NPZ Generation - Window 4096"
echo "========================================="
echo "Start time: $(date)"

# Modules
module purge
module load StdEnv/2023
module load python/3.11
module load arrow/21.0.0

# Verify PyArrow
python - <<'EOF'
import pyarrow as pa
print(f"PyArrow version: {pa.__version__}")
EOF

# Paths
PROJECT_DIR="/scratch/yuvraj17/SyntheticLogGeneration"
SCRIPT="$PROJECT_DIR/data_processing/parquet_to_windowed_npz_improved.py"
INPUT_DIR="/scratch/yuvraj17/enriched_parquet"
VOCAB_DIR="$PROJECT_DIR/dataset/metadata_all_events"
OUTPUT_DIR="/scratch/yuvraj17/improved_window_shards/windowed_npz_4096"

# Parameters
SEQ_LEN=4096
STRIDE=2048
WINDOWS_PER_SHARD=500   # Smaller shards for large windows
WORKERS=16              # Parallel processing

echo ""
echo "Configuration:"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Seq Length: $SEQ_LEN"
echo "  Stride: $STRIDE"
echo "  Windows per Shard: $WINDOWS_PER_SHARD"
echo "  Workers: $WORKERS"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "/scratch/yuvraj17/final_logs"

# Activate virtual environment
cd $SCRATCH/SyntheticLogGeneration
source .venv/bin/activate

# Set environment variables
export SCRATCH=$SCRATCH
export PYTHONUNBUFFERED=1

# Run conversion
echo "Starting conversion..."
python -u "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vocab-dir "$VOCAB_DIR" \
  --seq-len $SEQ_LEN \
  --stride $STRIDE \
  --windows-per-shard $WINDOWS_PER_SHARD \
  --workers $WORKERS

echo ""
echo "========================================="
echo "Conversion Complete!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "Output location: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Verify output with: ls -lh $OUTPUT_DIR/train/ | head -20"
echo "2. Test loader with: python test_improved_loader.py --data-dir $OUTPUT_DIR"
echo "3. Update training scripts to use improved loader"
echo "========================================="
