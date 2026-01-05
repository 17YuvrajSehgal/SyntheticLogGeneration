#!/bin/bash
#SBATCH --job-name=ramspeed_txt2parq
#SBATCH --account=def-naser2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-31
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# -----------------------------
# Modules (pyarrow is a module on Nibi)
# -----------------------------
module purge
module load StdEnv/2023
module load python/3.11
module load pyarrow/21.0.0

echo "===================================="
echo "Job:        $SLURM_JOB_NAME"
echo "JobID:      $SLURM_JOB_ID"
echo "ArrayTask:  $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start:      $(date)"
echo "Python:     $(python --version)"
python -c "import pyarrow; print('pyarrow:', pyarrow.__version__)"
echo "===================================="

# -----------------------------
# Paths (EDIT if needed)
# -----------------------------
IN_DIR="$SCRATCH/txt_traces_all_benchmarks/txt_traces_ramspeed"
OUT_DIR="$SCRATCH/parsed_parquet/ramspeed"
PY_SCRIPT="$HOME/projects/def-naser2/yuvraj/lttng-trace-jobs/txt_to_parquet_scripts/txt_to_parquet.py"     # <-- put your python script here (or change path)

mkdir -p "$OUT_DIR" logs

# -----------------------------
# Collect inputs and select this task's file
# -----------------------------
mapfile -t FILES < <(ls -1 "$IN_DIR"/ramspeed-all-events-run*.txt 2>/dev/null | sort)

NUM_FILES="${#FILES[@]}"
if [ "$NUM_FILES" -eq 0 ]; then
  echo "ERROR: No input files found in $IN_DIR (pattern: ramspeed-all-events-run*.txt)"
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if [ "$TASK_ID" -ge "$NUM_FILES" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID but only $NUM_FILES files exist."
  echo "TIP: change --array=0-$(($NUM_FILES-1))"
  exit 1
fi

IN_FILE="${FILES[$TASK_ID]}"
BASE="$(basename "$IN_FILE" .txt)"
RUN="${BASE##*-all-events-}"     # run0, run1, ...

OUT_FILE="$OUT_DIR/${RUN}.parquet"

echo "Input:  $IN_FILE"
echo "Output: $OUT_FILE"

# -----------------------------
# Run conversion
# -----------------------------
python "$PY_SCRIPT" \
  --input "$IN_FILE" \
  --output "$OUT_FILE" \
  --dataset "ramspeed" \
  --run-id "$RUN" \
  --batch-size 200000

echo "Done: $(date)"