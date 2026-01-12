#!/bin/bash
#SBATCH --job-name=learn-constraints
#SBATCH --account=def-naser2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ---- Environment ----
module --force purge
module load StdEnv/2023 python/3.11

source /project/6083487/yuvraj/SyntheticLogGeneration/.venv/bin/activate
cd /project/6083487/yuvraj/SyntheticLogGeneration

mkdir -p logs dataset

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"

# ---- Learn Constraints ----

# Use $SCRATCH directly or define a variable
REAL_GLOB="$SCRATCH/windowed_npz_256/**/*.npz"
OUTPUT="dataset/constraints_universal.json"

echo "Learning constraints from: $REAL_GLOB"
echo "Saving to: $OUTPUT"

python -u -m data_processing.learn_constraints \
  --real_glob "$REAL_GLOB" \
  --output "$OUTPUT" \
  --num_events 384 \
  --num_cpus 4 \
  --workers $SLURM_CPUS_PER_TASK

echo "Constraint learning completed."
