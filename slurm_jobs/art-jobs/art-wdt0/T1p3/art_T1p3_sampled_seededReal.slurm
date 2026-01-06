#!/bin/bash
#SBATCH --job-name=art-T1p3-samp-seeded
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

RUN_NAME="art-wdt0"
CKPT="$REPO/outputs/art_outputs/$RUN_NAME/checkpoints_ar/$BENCH/final_step_020000.pt"

NUM_SAMPLES=100000
BATCH_SAMPLES=128
SEQ_LEN=200
TEMP=1.3

RUN_OUT="$REPO/outputs/art_outputs/$RUN_NAME"
GENDIR="$RUN_OUT/generated_traces/ar/$BENCH/T1p3"
mkdir -p "$GENDIR"

OUTFILE="$GENDIR/synth_${NUM_SAMPLES}_T${TEMP}_sampled_seededReal.npz"

echo "========== ART SAMPLE =========="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "RUN:  $RUN_NAME"
echo "CKPT: $CKPT"
echo "OUT:  $OUTFILE"
nvidia-smi || true
echo "================================"

[[ -f "$CKPT" ]] || { echo "[ERROR] Missing checkpoint: $CKPT"; exit 1; }
[[ -d "$ROOT_SHARDS/$BENCH/train" ]] || echo "[WARN] Expected shard dir missing: $ROOT_SHARDS/$BENCH/train"

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
  --seed_from_real \
  --root_shards "$ROOT_SHARDS" \
  --benchmark "$BENCH" \
  --seed_split train \
  --max_seed_shards 10

python - <<PY
import numpy as np
d=np.load("$OUTFILE")
for k in ("event","dt","cpu"):
    a=d[k]
    print(k, a.shape, a.dtype, "min", int(a.min()), "max", int(a.max()))
PY