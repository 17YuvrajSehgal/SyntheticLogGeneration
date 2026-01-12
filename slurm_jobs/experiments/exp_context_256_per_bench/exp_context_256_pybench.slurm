#!/bin/bash
#SBATCH --job-name=pybench_exp_context_256
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/pybench/exp_context_256/exp_context_256_pybench/slurm_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/pybench/exp_context_256/exp_context_256_pybench/slurm_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

BENCHMARK="pybench"
RUN_NAME="exp_context_256_pybench"

# Put everything (including logs) inside the run folder
EXPERIMENT_DIR="$REPO/experiments_results/$BENCHMARK/exp_context_256/$RUN_NAME"
mkdir -p "$EXPERIMENT_DIR"

echo "[START] Experiment: $RUN_NAME"
echo "Benchmark: $BENCHMARK"
echo "Date: $(date)"
echo "Output Directory: $EXPERIMENT_DIR"

python -u train_experiment.py \
    --data-root "$SCRATCH/windowed_npz_256" \
    --benchmark "$BENCHMARK" \
    --seq-len 256 \
    --channels event dt cpu tid comm ret \
    --model-type diffusion \
    --d-model 512 \
    --nhead 8 \
    --num-layers 8 \
    --batch-size 256 \
    --num-cpus 4 \
    --tid-buckets 256 \
    --fd-cap 1025 \
    --epochs 100 \
    --steps 1000 \
    --lr 2e-4 \
    --mixed-precision bf16 \
    --log-dir "$EXPERIMENT_DIR" \
    --run-name "$RUN_NAME"

echo "[DONE] Finished"
