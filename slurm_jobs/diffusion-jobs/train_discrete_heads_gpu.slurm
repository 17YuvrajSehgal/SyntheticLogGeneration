#!/bin/bash
#SBATCH --job-name=disc-heads-full
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

source /project/6083487/yuvraj/SyntheticLogGeneration/.venv/bin/activate
cd /project/6083487/yuvraj/SyntheticLogGeneration

mkdir -p logs checkpoints_discrete generated_traces

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Submit dir: $(pwd)"
echo "Python: $(which python)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

echo "=== GPU at start ==="
nvidia-smi

# -------- GPU logger (runs in background) --------
GPU_LOG="logs/gpu-${SLURM_JOB_ID}.csv"
echo "timestamp,util_gpu,util_mem,mem_used_mb,mem_total_mb,power_w,temp_c" > "$GPU_LOG"

(
  while true; do
    ts=$(date +"%Y-%m-%d %H:%M:%S")
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits | awk -v t="$ts" -F',' '{gsub(/ /,""); print t","$1","$2","$3","$4","$5","$6}'
    sleep 10
  done
) >> "$GPU_LOG" &

GPU_LOG_PID=$!
echo "GPU logger PID: $GPU_LOG_PID"
echo "GPU log: $GPU_LOG"

# Ensure logger stops when job exits
trap "echo 'Stopping GPU logger'; kill $GPU_LOG_PID 2>/dev/null" EXIT

# -------- Training --------
python -u -m diffusion_model.train_discrete_heads \
  --benchmark compress-gzip \
  --steps 50000 \
  --batch_size 64 \
  --num_events 380 \
  --num_dt_buckets 256 \
  --num_cpus 4 \
  --save_every 500 \
  --out_dir checkpoints_discrete_full