#!/bin/bash
#SBATCH --job-name=diff-wcpu0
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/diffusion_logs/diff-wcpu0/slurm-%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/diffusion_logs/diff-wcpu0/slurm-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
BENCH="compress-gzip"
RUN_NAME="diff-wcpu0"

LOGDIR="$REPO/logs/diffusion_logs/$RUN_NAME"
OUTDIR="$REPO/outputs/diffusion_outputs/$RUN_NAME"
CKPT_DIR="$OUTDIR/checkpoints_discrete"

# Ensure dirs exist (Slurm won't create them)
mkdir -p "$LOGDIR" "$OUTDIR" "$CKPT_DIR"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "========== DIFFUSION TRAIN JOB START =========="
echo "RUN_NAME: $RUN_NAME"
echo "BENCH:    $BENCH"
echo "Time:     $(date)"
echo "Host:     $(hostname)"
echo "PWD:      $(pwd)"
echo "Repo:     $REPO"
echo "Python:   $(which python)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
echo "=== GPU at start ==="
nvidia-smi || true
echo "LOGDIR:   $LOGDIR"
echo "OUTDIR:   $OUTDIR"
echo "CKPT:     $CKPT_DIR"
echo "=============================================="

# -------- GPU logger (runs in background) --------
GPU_LOG="$LOGDIR/gpu-${SLURM_JOB_ID}.csv"
echo "timestamp,util_gpu,util_mem,mem_used_mb,mem_total_mb,power_w,temp_c" > "$GPU_LOG"

(
  while true; do
    ts=$(date +"%Y-%m-%d %H:%M:%S")
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits \
    | awk -v t="$ts" -F',' '{gsub(/ /,""); print t","$1","$2","$3","$4","$5","$6}'
    sleep 10
  done
) >> "$GPU_LOG" &

GPU_LOG_PID=$!
echo "GPU logger PID: $GPU_LOG_PID"
echo "GPU log: $GPU_LOG"

trap "echo 'Stopping GPU logger'; kill $GPU_LOG_PID 2>/dev/null || true" EXIT

# -------- Training --------
python -u -m diffusion_model.train_discrete_heads \
  --benchmark "$BENCH" \
  --root_shards dataset/window_shards \
  --steps 20000 \
  --batch_size 64 \
  --num_events 380 \
  --num_dt_buckets 256 \
  --num_cpus 4 \
  --w_cpu 0.0 \
  --save_every 500 \
  --out_dir "$CKPT_DIR" \
  2>&1 | tee "$LOGDIR/train.log"

echo "========== DIFFUSION TRAIN JOB DONE =========="
echo "Time: $(date)"
echo "Saved under: $OUTDIR"