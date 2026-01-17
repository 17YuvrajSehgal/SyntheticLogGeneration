#!/bin/bash
#SBATCH --job-name=ablation_ffmpeg
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/%u/ablation_ffmpeg_%j.out
#SBATCH --error=/scratch/%u/ablation_ffmpeg_%j.err

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "============================================================"
echo "Ablation Study Pipeline (Option 3)"
echo "============================================================"
echo "Benchmark: ffmpeg"
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

# Run the ablation pipeline
python run_ablation_pipeline.py \
    --benchmark ffmpeg \
    --base-epoch 19 \
    --system-epoch 19 \
    --full-epoch 19 \
    --num-samples 10000 \
    --max-parallel 9

echo ""
echo "============================================================"
echo "Ablation Pipeline Complete!"
echo "End: $(date)"
echo "============================================================"
echo ""
echo "Analyze results:"
echo "  python analyze_ablation_results.py --benchmark ffmpeg"
