#!/bin/bash
#SBATCH --job-name=art-T0p7-samp-free
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/art_logs/%x-%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/art_logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

BENCH="compress-gzip"
ROOT_SHARDS="dataset/window_shards"

RUN_NAME="art-wcpu0"  # change to art-wdt0
CKPT="$REPO/outputs/art_outputs/$RUN_NAME/checkpoints_ar/$BENCH/final_step_020000.pt"

NUM_SAMPLES=100000
BATCH_SAMPLES=128
SEQ_LEN=200
TEMP=1

GENDIR="$REPO/outputs/art_outputs/$RUN_NAME/generated_traces/ar/$BENCH/T0p7"
mkdir -p "$GENDIR"

OUTFILE="$GENDIR/synth_${NUM_SAMPLES}_T${TEMP}_sampled_freeStart.npz"

echo "[RUN] $RUN_NAME"
echo "CKPT: $CKPT"
echo "OUT : $OUTFILE"
[[ -f "$CKPT" ]] || { echo "[ERROR] Missing checkpoint: $CKPT"; exit 1; }

python -u -m autoregressive_transformer.sample_autoregressive \
  --ckpt "$CKPT" \
  --out "$OUTFILE" \
  --num_samples "$NUM_SAMPLES" \
  --batch_samples "$BATCH_SAMPLES" \
  --seq_len "$SEQ_LEN" \
  --d_model 256 \
  --nhead 8 \
  --num_layers 6 \
  --num_events 380 \
  --num_dt_buckets 256 \
  --num_cpus 4 \
  --temperature "$TEMP" \
  --root_shards "$ROOT_SHARDS" \
  --benchmark "$BENCH"