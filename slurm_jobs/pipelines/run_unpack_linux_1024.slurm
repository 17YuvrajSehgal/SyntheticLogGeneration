#!/bin/bash
#SBATCH --job-name=pipeline_unpack-linux_1024
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/pipeline_unpack-linux_1024_%j.out
#SBATCH --error=/scratch/%u/pipeline_unpack-linux_1024_%j.err

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "============================================================"
echo "Automated Experiment Pipeline"
echo "============================================================"
echo "Benchmark: unpack-linux"
echo "Window Size: 1024"
echo "JobID: $SLURM_JOB_ID"
echo "Node:  $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

# Show GPU info
nvidia-smi

# Activate virtual environment
cd $SCRATCH/SyntheticLogGeneration
source .venv/bin/activate

# Set environment variables
export SCRATCH=$SCRATCH
export PYTHONUNBUFFERED=1

# Run the pipeline
python run_pipeline.py \
    --benchmark unpack-linux \
    --window 1024 \
    --checkpoint-epoch 1 \
    --num-samples 10000

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "End: $(date)"
echo "============================================================"
