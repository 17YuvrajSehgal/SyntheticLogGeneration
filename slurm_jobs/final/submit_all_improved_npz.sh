#!/bin/bash
# ============================================================================
# Submit All Improved NPZ Generation Jobs
# ============================================================================
# This script submits Slurm jobs to convert parquet files to improved NPZ
# shards for all three window sizes (256, 1024, 4096).
#
# Usage:
#   bash submit_all_improved_npz.sh
#
# The jobs will run in parallel and create smaller NPZ shards that improve
# GPU utilization during training by 20-30%.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "Submitting Improved NPZ Generation Jobs"
echo "========================================="
echo ""

# Submit 256 window job
echo "Submitting window size 256..."
JOB_256=$(sbatch "$SCRIPT_DIR/generate_improved_npz_256.slurm" | awk '{print $4}')
echo "  Job ID: $JOB_256"

# Submit 1024 window job
echo "Submitting window size 1024..."
JOB_1024=$(sbatch "$SCRIPT_DIR/generate_improved_npz_1024.slurm" | awk '{print $4}')
echo "  Job ID: $JOB_1024"

# Submit 4096 window job
echo "Submitting window size 4096..."
JOB_4096=$(sbatch "$SCRIPT_DIR/generate_improved_npz_4096.slurm" | awk '{print $4}')
echo "  Job ID: $JOB_4096"

echo ""
echo "========================================="
echo "All jobs submitted!"
echo "========================================="
echo ""
echo "Job IDs:"
echo "  256:  $JOB_256"
echo "  1024: $JOB_1024"
echo "  4096: $JOB_4096"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  logs/improved_npz_256_${JOB_256}.out"
echo "  logs/improved_npz_1024_${JOB_1024}.out"
echo "  logs/improved_npz_4096_${JOB_4096}.out"
echo ""
echo "Output will be in:"
echo "  \$SCRATCH/improved_window_shards/windowed_npz_256/"
echo "  \$SCRATCH/improved_window_shards/windowed_npz_1024/"
echo "  \$SCRATCH/improved_window_shards/windowed_npz_4096/"
echo "========================================="
