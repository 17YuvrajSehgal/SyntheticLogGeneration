#!/bin/bash
#SBATCH --job-name=gpu_hold_6h
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/gpu_hold_slurm_%x_%j.out
#SBATCH --error=/scratch/%u/gpu_hold_slurm_%x_%j.err

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "JobID: $SLURM_JOB_ID"
echo "Node:  $SLURM_NODELIST"
echo "Start: $(date)"
echo "Holding allocation for 6 hours..."
nvidia-smi || true

# keep the allocation alive
sleep 6h
