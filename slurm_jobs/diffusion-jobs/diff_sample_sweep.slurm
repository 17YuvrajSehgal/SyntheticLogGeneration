#!/bin/bash
#SBATCH --job-name=diff-sweep
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/diffusion_logs/%x-%j.out
#SBATCH --error=/project/def-naser2/yuvraj/SyntheticLogGeneration/logs/diffusion_logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="/project/def-naser2/yuvraj/SyntheticLogGeneration"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

BENCH="compress-gzip"

# Which trained diffusion runs to sample from (folders under outputs/diffusion_outputs/)
# Example from your screenshot: diff-wcpu0, diff-wcpu2, diff-wdt0
DIFF_RUNS=(
  "diff-wcpu0"
  "diff-wcpu2"
  "diff-wdt0"
)

# Keep T consistent with training unless you explicitly trained with a different T
T=1000
SEQ_LEN=200

# Compare variance across random seeds
SEEDS=(123 456 789)

# Sample sizes you want to compare quickly vs final
# (Use 10k/20k for sweeps; later do 100k for the finalists)
SIZES=(10000 20000)
BATCH_SAMPLES=512

pick_best_ckpt () {
  local ckptdir="$1"
  local best=""
  best="$(ls -1 "$ckptdir"/best_step_*.pt 2>/dev/null | tail -n 1 || true)"
  if [[ -z "$best" ]]; then best="$(ls -1 "$ckptdir"/step_*.pt 2>/dev/null | tail -n 1 || true)"; fi
  if [[ -z "$best" ]]; then best="$(ls -1 "$ckptdir"/final_step_*.pt 2>/dev/null | tail -n 1 || true)"; fi
  echo "$best"
}

echo "========== DIFFUSION SAMPLING SWEEP =========="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO"
echo "GPU:"
nvidia-smi || true
echo "==============================================="

for RUN_NAME in "${DIFF_RUNS[@]}"; do
  RUN_OUT="$REPO/outputs/diffusion_outputs/$RUN_NAME"
  RUN_LOG="$REPO/logs/diffusion_logs/$RUN_NAME"
  mkdir -p "$RUN_OUT" "$RUN_LOG"

  CKPTDIR="$RUN_OUT/checkpoints_discrete/$BENCH"
  if [[ ! -d "$CKPTDIR" ]]; then
    echo "[SKIP] Missing checkpoint dir: $CKPTDIR"
    continue
  fi

  CKPT="$(pick_best_ckpt "$CKPTDIR")"
  if [[ -z "$CKPT" ]]; then
    echo "[SKIP] No checkpoint found in: $CKPTDIR"
    ls -lh "$CKPTDIR" || true
    continue
  fi

  echo
  echo "[RUN] $RUN_NAME"
  echo "  ckpt: $CKPT"

  GENDIR="$RUN_OUT/generated_traces/discrete/$BENCH"
  mkdir -p "$GENDIR"

  for n in "${SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUTFILE="$GENDIR/synth_${n}_T${T}_seed${seed}.npz"
      echo "  -> $OUTFILE"

      python -u -m diffusion_model.sample_synthetic \
        --ckpt "$CKPT" \
        --out "$OUTFILE" \
        --num_samples "$n" \
        --batch_samples "$BATCH_SAMPLES" \
        --seq_len "$SEQ_LEN" \
        --d_model 256 \
        --T "$T" \
        --num_events 380 \
        --num_dt_buckets 256 \
        --num_cpus 4 \
        --seed "$seed"

      python - <<PY
import numpy as np
p = "$OUTFILE"
d = np.load(p)
for k in ("event","dt","cpu"):
    a = d[k]
    print("   ", k, a.shape, a.dtype, "min", int(a.min()), "max", int(a.max()))
PY

    done
  done
done

echo "========== DONE =========="
echo "Time: $(date)"