#!/bin/bash
#SBATCH --job-name=build_vocab_all
#SBATCH --account=def-naser2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/build_vocab_all_%j.out
#SBATCH --error=logs/build_vocab_all_%j.err

set -euo pipefail

PROJECT_DIR="$SCRATCH/SyntheticLogGeneration"
SCRIPT="$PROJECT_DIR/dataset/build_vocab.py"

ROOT_DIR="$SCRATCH/txt_traces_all_benchmarks"
OUT_DIR="$PROJECT_DIR/dataset/metadata_all_events"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUT_DIR"

cd "$PROJECT_DIR"

python -u "$SCRIPT" \
  --root "$ROOT_DIR" \
  --glob "*-all-events-run*.txt" \
  --out_dir "$OUT_DIR" \
  --min_count 1 \
  --sort freq