#!/bin/bash
#SBATCH --job-name=privacy-100k-fast
#SBATCH --account=def-naser2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module --force purge
module load StdEnv/2023 python/3.11

# Activate your venv
source /project/6083487/yuvraj/SyntheticLogGeneration/.venv/bin/activate

# Go to repo
cd /project/6083487/yuvraj/SyntheticLogGeneration

# Make sure logs + output dir exist
mkdir -p logs evaluation_metrics/privacy_analysis_full_100k_fast

# Helpful runtime env vars
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started on: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"

python -u evaluation_metrics/privacy_analysis.py \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --synth "art_outputs/generated_traces/ar/compress-gzip/synth_100k.npz" \
  --max_real_windows 30000 \
  --max_synth_windows 100000 \
  --distance_metric euclidean \
  --check_synth_diversity \
  --output_dir evaluation_metrics/privacy_analysis_full_100k_fast \
  --seed 42

echo "Job finished on: $(date)"
ls -lh evaluation_metrics/privacy_analysis_full_100k_fast || true
