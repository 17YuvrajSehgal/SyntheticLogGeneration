#!/bin/bash
#SBATCH --job-name=real_baseline_unpack-linux_4096
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --output=/scratch/%u/real_baseline_unpack-linux_4096_%j.out
#SBATCH --error=/scratch/%u/real_baseline_unpack-linux_4096_%j.err

set -euo pipefail

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "============================================================"
echo "Downstream Predictor Training (REAL baseline): unpack-linux-4096"
echo "============================================================"
echo "JobID:  $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "============================================================"

# GPU info
command -v nvidia-smi &>/dev/null && nvidia-smi || true

# Activate environment
cd "$SCRATCH/SyntheticLogGeneration"
source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "Starting training..."
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/unpack-linux/4096/data/real_train.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/unpack-linux/4096/data/real_test.npz" --run-name "real_baseline_stream_4096" --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/unpack-linux/4096/results" --seq-len 128 --batch-size 64 --epochs 20


echo ""
echo "============================================================"
echo "Training Complete!"
echo "End: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
