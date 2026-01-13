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

out_dir = Path("slurm_jobs/experiments/exp_context_256_per_bench")
out_dir.mkdir(parents=True, exist_ok=True)

# Where run folders (and logs) should live
repo_root = Path("/project/def-naser2/yuvraj/SyntheticLogGeneration")
exp_root = repo_root / "experiments_results"

template = """#!/bin/bash
#SBATCH --job-name={BENCHMARK}_exp_context_256
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/{BENCHMARK}/exp_context_256/exp_context_256_{BENCHMARK}/slurm_%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/experiments_results/{BENCHMARK}/exp_context_256/exp_context_256_{BENCHMARK}/slurm_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

BENCHMARK="{BENCHMARK}"
RUN_NAME="exp_context_256_{BENCHMARK}"

# Put everything (including logs) inside the run folder
EXPERIMENT_DIR="$REPO/experiments_results/$BENCHMARK/exp_context_256/$RUN_NAME"
mkdir -p "$EXPERIMENT_DIR"

echo "[START] Experiment: $RUN_NAME"
echo "Benchmark: $BENCHMARK"
echo "Date: $(date)"
echo "Output Directory: $EXPERIMENT_DIR"

python -u train_experiment.py \\
    --data-root "$SCRATCH/windowed_npz_256" \\
    --benchmark "$BENCHMARK" \\
    --seq-len 256 \\
    --channels event dt cpu tid comm ret \\
    --model-type diffusion \\
    --d-model 512 \\
    --nhead 8 \\
    --num-layers 8 \\
    --batch-size 256 \\
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
    run_dir = exp_root / bench / "exp_context_256" / f"exp_context_256_{bench}"
    run_dir.mkdir(parents=True, exist_ok=True)

    slurm_text = template.format(BENCHMARK=bench)
    slurm_path = out_dir / f"exp_context_256_{bench}.slurm"
    slurm_path.write_text(slurm_text)
    print(f"[OK] wrote {slurm_path} (logs in {run_dir})")
