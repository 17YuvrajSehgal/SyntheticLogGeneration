#!/bin/bash
#SBATCH --job-name=zenodo_dl
#SBATCH --account=def-naser2        # <-- change if needed
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/zenodo_dl_%j.out
#SBATCH --error=logs/zenodo_dl_%j.err

set -euo pipefail

echo "===================================="
echo " Zenodo download job started"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Time: $(date)"
echo "===================================="

# ------------------------------
# Directories
# ------------------------------
SCRATCH_DIR="$SCRATCH/zenodo_benchmarks"
LOG_DIR="$HOME/logs"

mkdir -p "$SCRATCH_DIR"
mkdir -p "$LOG_DIR"

cd "$SCRATCH_DIR"

# ------------------------------
# URLs
# ------------------------------
URLS=(
  "https://zenodo.org/records/437170/files/phpbench.zip?download=1"
  "https://zenodo.org/records/437170/files/pybench.zip?download=1"
  "https://zenodo.org/records/437170/files/ramspeed.zip?download=1"
  "https://zenodo.org/records/437170/files/scimark2.zip?download=1"
  "https://zenodo.org/records/437170/files/stream.zip?download=1"
  "https://zenodo.org/records/437170/files/unpack-linux.zip?download=1"
)

# ------------------------------
# Download loop
# ------------------------------
for url in "${URLS[@]}"; do
  fname=$(basename "${url%%\?*}")
  echo "Downloading $fname ..."
  wget -c -O "$fname" "$url"
done

echo "===================================="
echo " All downloads completed"
echo " Files in: $SCRATCH_DIR"
echo " Time: $(date)"
echo "===================================="
