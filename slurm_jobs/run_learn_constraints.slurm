#!/bin/bash
#SBATCH --job-name=learn-constraints
#SBATCH --account=def-naser2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
# NOTE: Update glob patterns if you want to learn from all benchmarks or a specific one.
# Here we default to compress-gzip as an example.

python -u -m synthetic_data_processing.learn_constraints \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --output "dataset/constraints.json" \
  --num_events 384 \
  --num_dt_buckets 256 \
  --num_cpus 4

echo "Constraint learning completed."
