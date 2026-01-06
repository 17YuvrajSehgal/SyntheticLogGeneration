#!/bin/bash
#SBATCH --job-name=art-sweep
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

# How many windows to generate per config
NUM_SAMPLES=20000
BATCH_SAMPLES=128
SEQ_LEN=200

# Which trained AR runs to sample from (folders under outputs/art_outputs/)
# Example from your screenshot: art-wcpu0, art-wdt0
AR_RUNS=(
  "art-wcpu0"
  "art-wdt0"
)

# Sampling settings to compare
TEMPS=(0.7 1.0 1.3)
SEEDS=(123 456)

# Two modes:
#  - greedy=1 (deterministic argmax)
#  - greedy=0 (stochastic categorical sampling)
GREEDY_FLAGS=(0 1)

# Two seeding modes:
#  - seed_from_real=1 uses first token triplets from real shards (more realistic starts)
#  - seed_from_real=0 starts from zeros (stress-test model free-running)
SEED_FROM_REAL_FLAGS=(1 0)

# ---- helpers ----
pick_best_ckpt () {
  local ckptdir="$1"
  local best=""
  best="$(ls -1 "$ckptdir"/best_step_*.pt 2>/dev/null | tail -n 1 || true)"
  if [[ -z "$best" ]]; then best="$(ls -1 "$ckptdir"/step_*.pt 2>/dev/null | tail -n 1 || true)"; fi
  if [[ -z "$best" ]]; then best="$(ls -1 "$ckptdir"/final_step_*.pt 2>/dev/null | tail -n 1 || true)"; fi
  echo "$best"
}

echo "========== ART SAMPLING SWEEP =========="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO"
echo "GPU:"
nvidia-smi || true
echo "========================================"

for RUN_NAME in "${AR_RUNS[@]}"; do
  RUN_OUT="$REPO/outputs/art_outputs/$RUN_NAME"
  RUN_LOG="$REPO/logs/art_logs/$RUN_NAME"
  mkdir -p "$RUN_OUT" "$RUN_LOG"

  CKPTDIR="$RUN_OUT/checkpoints_ar/$BENCH"
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

  # Where generated files go for THIS trained run
  GENDIR="$RUN_OUT/generated_traces/ar/$BENCH"
  mkdir -p "$GENDIR"

  for temp in "${TEMPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for greedy in "${GREEDY_FLAGS[@]}"; do
        for sreal in "${SEED_FROM_REAL_FLAGS[@]}"; do

          mode_g="sampled"
          [[ "$greedy" == "1" ]] && mode_g="greedy"

          mode_s="seededReal"
          [[ "$sreal" == "0" ]] && mode_s="freeStart"

          OUTFILE="$GENDIR/synth_${NUM_SAMPLES}_T${temp}_seed${seed}_${mode_g}_${mode_s}.npz"
          echo "  -> $OUTFILE"

          args=(
            -u -m autoregressive_transformer.sample_autoregressive
            --ckpt "$CKPT"
            --out "$OUTFILE"
            --num_samples "$NUM_SAMPLES"
            --batch_samples "$BATCH_SAMPLES"
            --seq_len "$SEQ_LEN"
            --d_model 256
            --nhead 8
            --num_layers 6
            --num_events 380
            --num_dt_buckets 256
            --num_cpus 4
            --temperature "$temp"
            --seed "$seed"
            --root_shards "$ROOT_SHARDS"
            --benchmark "$BENCH"
          )

          if [[ "$greedy" == "1" ]]; then
            args+=(--greedy)
          fi
          if [[ "$sreal" == "1" ]]; then
            args+=(--seed_from_real --seed_split train --max_seed_shards 10)
          fi

          python "${args[@]}"

          # quick sanity check (fast)
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
  done
done

echo "========== DONE =========="
echo "Time: $(date)"