#!/bin/bash
#SBATCH --job-name=diff-wcpu2-T0250-n20k
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

# ---- config (edit these per file/folder) ----
RUN_NAME="diff-wcpu2"
T=250
# --------------------------------------------

SEQ_LEN=200
D_MODEL=256
NUM_SAMPLES=20000
BATCH_SAMPLES=128

CKPT="$REPO/outputs/diffusion_outputs/$RUN_NAME/checkpoints_discrete/$BENCH/final_step_020000.pt"

GENDIR="$REPO/outputs/diffusion_outputs/$RUN_NAME/generated_traces/discrete/$BENCH"
mkdir -p "$GENDIR"

# Safer filename (won’t accidentally overwrite if you change batch size or reuse scripts)
OUTFILE="$GENDIR/${RUN_NAME}_synth_${NUM_SAMPLES}_L${SEQ_LEN}_T${T}_bs${BATCH_SAMPLES}.npz"

echo "========== DIFF SAMPLE =========="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "RUN : $RUN_NAME"
echo "CKPT: $CKPT"
echo "OUT : $OUTFILE"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi || true
echo "================================="

[[ -f "$CKPT" ]] || { echo "[ERROR] Missing checkpoint: $CKPT"; exit 1; }

python -u -m diffusion_model.sample_synthetic \
  --ckpt "$CKPT" \
  --out "$OUTFILE" \
  --num_samples "$NUM_SAMPLES" \
  --batch_samples "$BATCH_SAMPLES" \
  --seq_len "$SEQ_LEN" \
  --d_model "$D_MODEL" \
  --T "$T" \
  --num_events 380 \
  --num_dt_buckets 256 \
  --num_cpus 4

python - <<PY
import numpy as np
d=np.load("$OUTFILE")
for k in ("event","dt","cpu"):
    a=d[k]
    print(k, a.shape, a.dtype, "min", int(a.min()), "max", int(a.max()))
PY