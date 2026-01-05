#!/bin/bash
#SBATCH --job-name=art-wcpu0
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=60:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/art_logs/art-wcpu0/slurm-%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/art_logs/art-wcpu0/slurm-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
BENCH="compress-gzip"
RUN_NAME="art-wcpu0"

LOGDIR="$REPO/logs/art_logs/$RUN_NAME"
OUTDIR="$REPO/outputs/art_outputs/$RUN_NAME"
CKPT_BASE="$OUTDIR/checkpoints_ar"

# Ensure dirs exist (Slurm won't create them)
mkdir -p "$LOGDIR" "$OUTDIR" "$CKPT_BASE"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "========== ART TRAIN JOB START =========="
echo "RUN_NAME: $RUN_NAME"
echo "BENCH:    $BENCH"
echo "Time:     $(date)"
echo "Host:     $(hostname)"
echo "PWD:      $(pwd)"
echo "Repo:     $REPO"
echo "Python:   $(which python)"
python -V
nvidia-smi || true
echo "LOGDIR:   $LOGDIR"
echo "OUTDIR:   $OUTDIR"
echo "CKPT:     $CKPT_BASE"
echo "========================================"

python -u -m autoregressive_transformer.train_autoregressive \
  --benchmark "$BENCH" \
  --root_shards dataset/window_shards \
  --steps 20000 \
  --batch_size 64 \
  --lr 2e-4 \
  --seq_len 200 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 6 \
  --dropout 0.1 \
  --num_events 380 \
  --num_dt_buckets 256 \
  --num_cpus 4 \
  --num_workers 4 \
  --save_every 2000 \
  --eval_every 2000 \
  --w_cpu 0.0 \
  --log_acc \
  --out_dir "$CKPT_BASE" \
  2>&1 | tee "$LOGDIR/train.log"

echo "[INFO] Checkpoints:"
ls -lh "$CKPT_BASE/$BENCH" || true

echo "========== ART TRAIN JOB DONE =========="
echo "Time: $(date)"