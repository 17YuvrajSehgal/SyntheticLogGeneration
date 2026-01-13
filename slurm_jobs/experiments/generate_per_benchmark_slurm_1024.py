#!/usr/bin/env python3
from pathlib import Path

benchmarks = [
    "compress-gzip",
    "ffmpeg",
    "iozone",
    "phpbench",
    "pybench",
    "ramspeed",
    "scimark2",
    "stream",
    "unpack-linux",
]

out_dir = Path("slurm_jobs/experiments/exp_context_1024_per_bench")
out_dir.mkdir(parents=True, exist_ok=True)

# Repo + experiment roots (used to pre-create log dirs)
repo_root = Path("/project/def-naser2/yuvraj/SyntheticLogGeneration")
exp_root = repo_root / "experiments_results"

template = """#!/bin/bash
#SBATCH --job-name={BENCHMARK}_exp_context_1024
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/{BENCHMARK}/exp_context_1024/exp_context_1024_{BENCHMARK}/slurm_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/{BENCHMARK}/exp_context_1024/exp_context_1024_{BENCHMARK}/slurm_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

BENCHMARK="{BENCHMARK}"
RUN_NAME="exp_context_1024_{BENCHMARK}"

# Put everything (including logs) inside the run folder
EXPERIMENT_DIR="$REPO/experiments_results/$BENCHMARK/exp_context_1024/$RUN_NAME"
mkdir -p "$EXPERIMENT_DIR"

echo "[START] Experiment: $RUN_NAME"
echo "Benchmark: $BENCHMARK"
echo "Date: $(date)"
echo "Output Directory: $EXPERIMENT_DIR"

python -u train_experiment.py \\
    --data-root "$SCRATCH/windowed_npz_1024" \\
    --benchmark "$BENCHMARK" \\
    --seq-len 1024 \\
    --channels event dt cpu tid comm ret \\
    --model-type diffusion \\
    --d-model 512 \\
    --nhead 8 \\
    --num-layers 8 \\
    --batch-size 128 \\
    --num-workers 8 \\
    --num-cpus 4 \\
    --tid-buckets 256 \\
    --fd-cap 1025 \\
    --epochs 20 \\
    --steps 1000 \\
    --lr 2e-4 \\
    --max-steps-per-epoch 500 \\
    --mixed-precision bf16 \\
    --log-dir "$EXPERIMENT_DIR" \\
    --run-name "$RUN_NAME"

echo "[DONE] Finished"
"""

for bench in benchmarks:
    # Pre-create the run directory so Slurm can open stdout/stderr files
    run_dir = exp_root / bench / "exp_context_1024" / f"exp_context_1024_{bench}"
    run_dir.mkdir(parents=True, exist_ok=True)

    slurm_text = template.format(BENCHMARK=bench)
    slurm_path = out_dir / f"exp_context_1024_{bench}.slurm"
    slurm_path.write_text(slurm_text)
    print(f"[OK] wrote {slurm_path} (logs in {run_dir})")
