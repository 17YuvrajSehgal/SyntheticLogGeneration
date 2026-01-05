#!/bin/bash
#SBATCH --job-name=sample-synth
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

source /project/6083487/yuvraj/SyntheticLogGeneration/.venv/bin/activate
cd /project/6083487/yuvraj/SyntheticLogGeneration

mkdir -p logs generated_traces/discrete/compress-gzip

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "Python: $(which python)"
nvidia-smi

# ---- Sampling from your TEST checkpoint ----
python -u -m diffusion_model.sample_synthetic \
  --ckpt checkpoints_discrete_full/compress-gzip/final_step_050000.pt \
  --out generated_traces/discrete/compress-gzip/synth_2k_step50k_FIXED.npz \
  --num_samples 2000 \
  --seq_len 200 \
  --d_model 256 \
  --T 1000 \
  --num_events 380 \
  --num_dt_buckets 256 \
  --num_cpus 4

echo "Done. Output:"
ls -lh generated_traces/discrete/compress-gzip/synth_2k_step2k_FIXED.npz