#!/bin/bash
#SBATCH --job-name=unzip_bench
#SBATCH --account=def-naser2        # change if needed
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/unzip_%j.out
#SBATCH --error=logs/unzip_%j.err

set -euo pipefail

echo "===================================="
echo " Unzip job started"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Time: $(date)"
echo "===================================="

# ------------------------------
# Directories
# ------------------------------
ZIP_DIR="$SCRATCH/zenodo_benchmarks"
OUT_DIR="$SCRATCH/unzipped-benchmarks"

mkdir -p "$OUT_DIR"
cd "$ZIP_DIR"

# ------------------------------
# Unzip loop
# ------------------------------
for zipfile in *.zip; do
  name=$(basename "$zipfile" .zip)
  dest="$OUT_DIR/$name"

  if [[ -d "$dest" ]]; then
    echo "[SKIP] $name already exists"
    continue
  fi

  echo "------------------------------------"
  echo "Unzipping $zipfile -> $dest"
  mkdir -p "$dest"
  unzip -q "$zipfile" -d "$dest"
done

echo "===================================="
echo " All files unzipped"
echo " Output directory:"
echo "   $OUT_DIR"
echo " Time: $(date)"
echo "===================================="