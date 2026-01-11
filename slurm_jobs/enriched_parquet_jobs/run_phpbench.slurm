#!/bin/bash
#SBATCH --job-name=parquet_phpbench
#SBATCH --account=def-naser2
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/parquet_phpbench_%j.out
#SBATCH --error=logs/parquet_phpbench_%j.err

set -euo pipefail

# -----------------------------
# Modules (pyarrow is a module on Nibi)
# -----------------------------
module purge
module load StdEnv/2023
module load python/3.11
module load pyarrow/21.0.0

# Paths
PROJECT_DIR="/project/def-naser2/yuvraj/SyntheticLogGeneration"
SCRIPT="$PROJECT_DIR/dataset/txt_to_enriched_parquet.py"
VOCAB="$PROJECT_DIR/dataset/metadata_all_events/vocab.json"

INPUT_DIR="$SCRATCH/txt_traces_all_benchmarks/txt_traces_phpbench"
OUTPUT_DIR="$SCRATCH/enriched_parquet/txt_traces_phpbench"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

echo "[INFO] Processing txt_traces_phpbench..."

# Iterate over all txt files in the directory
# Using GNU parallel would be faster but serial is safer for memory.
# We have ~32 files. 4 hours should be enough.

for file in "$INPUT_DIR"/*.txt; do
    [ -e "$file" ] || continue
    
    filename=$(basename "$file")
    stub="${filename%.txt}"
    outfile="$OUTPUT_DIR/${stub}.parquet"
    
    echo "[INFO] Converting $filename -> $outfile"
    
    python -u "$SCRIPT" \
      --input "$file" \
      --output "$outfile" \
      --vocab "$VOCAB"
      
done

echo "[INFO] All files in txt_traces_phpbench processed."
