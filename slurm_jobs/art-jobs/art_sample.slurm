#!/bin/bash
#SBATCH --job-name=art-sample
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/art_logs/%x-%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/art_logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

# ---- ALWAYS hard-set repo path on Nibi (same as train script) ----
REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"

# Slurm does NOT create output dirs
mkdir -p "$REPO/art_logs"

cd "$REPO"
source "$REPO/.venv/bin/activate"

LOGDIR="$REPO/art_logs"
OUTDIR="$REPO/art_outputs"

CKPT_BASE="$OUTDIR/checkpoints_ar"     # base dir (no benchmark)
BENCH="compress-gzip"
CKPTDIR="$CKPT_BASE/$BENCH"

GENDIR="$OUTDIR/generated_traces/ar/$BENCH"
mkdir -p "$LOGDIR" "$GENDIR"

export PYTHONUNBUFFERED=1

echo "========== ART SAMPLE JOB START =========="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "Repo: $REPO"
echo "Python: $(which python)"
python -V
nvidia-smi || true
echo "========================================="

# ---- Select checkpoint (best > step > final) ----
BEST_CKPT="$(ls -1 "$CKPTDIR"/best_step_*.pt 2>/dev/null | tail -n 1 || true)"
if [[ -z "${BEST_CKPT}" ]]; then
  BEST_CKPT="$(ls -1 "$CKPTDIR"/step_*.pt 2>/dev/null | tail -n 1 || true)"
fi
if [[ -z "${BEST_CKPT}" ]]; then
  BEST_CKPT="$(ls -1 "$CKPTDIR"/final_step_*.pt 2>/dev/null | tail -n 1 || true)"
fi

if [[ -z "${BEST_CKPT}" ]]; then
  echo "[ERROR] No checkpoint found in: $CKPTDIR"
  echo "Contents:"
  ls -lh "$CKPTDIR" || true
  exit 1
fi

echo "[INFO] Using checkpoint:"
echo "       $BEST_CKPT"

# ---- Sample ----
python -u -m autoregressive_transformer.sample_autoregressive --ckpt "$BEST_CKPT" --out "$GENDIR/synth_100k.npz" --num_samples 100000 --batch_samples 64 --seq_len 200 --d_model 256 --nhead 8 --num_layers 6 --num_events 380 --num_dt_buckets 256 --num_cpus 4 --temperature 1.0 --seed_from_real --root_shards "dataset/window_shards" --benchmark "$BENCH" --seed_split train --max_seed_shards 10

echo "[INFO] Generated file:"
ls -lh "$GENDIR/synth_200_sanity.npz"

echo "[INFO] Quick range check:"
python - << PY
import numpy as np
p = "$GENDIR/synth_200_sanity.npz"
d = np.load(p)
for k in d.files:
    a = d[k]
    print(k, a.shape, a.dtype, "min", int(a.min()), "max", int(a.max()))
PY

echo "========== ART SAMPLE JOB DONE =========="
echo "Time: $(date)"