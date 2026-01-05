#!/bin/bash
#SBATCH --job-name=sample-synth-50k
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

source /project/6083487/yuvraj/SyntheticLogGeneration/.venv/bin/activate
cd /project/6083487/yuvraj/SyntheticLogGeneration

mkdir -p logs generated_traces/discrete/compress-gzip
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
nvidia-smi

# python -u -m diffusion_model.sample_synthetic \
#   --ckpt checkpoints_discrete_full/compress-gzip/step_050000.pt \
#   --out generated_traces/discrete/compress-gzip/synth_100k_step50k.npz \
#   --num_samples 100000 \
#   --batch_samples 512
#   --seq_len 200 \
#   --d_model 256 \
#   --T 1000 \
#   --num_events 380 \
#   --num_dt_buckets 256 \
#   --num_cpus 4
  
python -u -m diffusion_model.sample_synthetic \
    --ckpt checkpoints_discrete_full/compress-gzip/step_050000.pt \
    --out generated_traces/discrete/compress-gzip/synth_100k_step50k.npz \
    --num_samples 100000 \
    --batch_samples 512 \
    --seq_len 200 \
    --T 1000 \
    --num_events 380 \
    --num_dt_buckets 256 \
    --num_cpus 4


ls -lh generated_traces/discrete/compress-gzip/synth_100k_step50k.npz
