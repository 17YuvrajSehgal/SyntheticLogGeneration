#!/bin/bash
#SBATCH --job-name=art-train
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=60:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/art_logs/%x-%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/art_logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

# ---- ALWAYS use the real repo path on Nibi (do NOT derive from $0) ----
REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"

# Make sure log dir exists (Slurm won't create it)
mkdir -p "$REPO/art_logs"

cd "$REPO"

# Activate venv from repo (now path is correct)
source "$REPO/.venv/bin/activate"

LOGDIR="$REPO/art_logs"
OUTDIR="$REPO/art_outputs"
CKPT_BASE="$OUTDIR/checkpoints_ar"   # base dir (NO benchmark suffix)
BENCH="compress-gzip"

mkdir -p "$LOGDIR" "$OUTDIR" "$CKPT_BASE"
export PYTHONUNBUFFERED=1

echo "========== ART TRAIN JOB START =========="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "Repo: $REPO"
echo "Python: $(which python)"
python -V
nvidia-smi || true
echo "========================================="

python -u -m autoregressive_transformer.train_autoregressive \
  --benchmark "$BENCH" \
  --root_shards window_shards \
  --steps 50000 \
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
  --w_cpu 2.0 \
  --log_acc \
  --out_dir "$CKPT_BASE"


echo "[INFO] Checkpoints:"
ls -lh "$CKPT_BASE/$BENCH" || true

echo "========== ART TRAIN JOB DONE =========="
echo "Time: $(date)"