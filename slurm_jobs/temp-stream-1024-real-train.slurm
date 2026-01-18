#!/bin/bash
#SBATCH --job-name=real_baseline_stream_1024
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/real_baseline_stream_1024_%j.out
#SBATCH --error=/scratch/%u/real_baseline_stream_1024_%j.err

set -euo pipefail

# Load modules
module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

echo "============================================================"
echo "Downstream Predictor Training (REAL baseline): stream-1024"
echo "============================================================"
echo "JobID:  $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "============================================================"

# GPU info
command -v nvidia-smi &>/dev/null && nvidia-smi || true

# Activate environment
cd "$SCRATCH/SyntheticLogGeneration"
source .venv/bin/activate

export PYTHONUNBUFFERED=1

# Configuration
BENCHMARK="stream"
CONTEXT_LEN=1024
SEQ_LEN=128
BATCH_SIZE=64
EPOCHS=20

TRAIN_DATA="$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/${BENCHMARK}/${CONTEXT_LEN}/data/real_train.npz"
TEST_DATA="$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/${BENCHMARK}/${CONTEXT_LEN}/data/real_test.npz"
OUTPUT_DIR="$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/${BENCHMARK}/${CONTEXT_LEN}/results"
RUN_NAME="real_baseline_${BENCHMARK}_${CONTEXT_LEN}"

echo ""
echo "Configuration:"
echo "  Benchmark:     $BENCHMARK"
echo "  Context Len:   $CONTEXT_LEN"
echo "  Seq Len:       $SEQ_LEN"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Epochs:        $EPOCHS"
echo "  Train Data:    $TRAIN_DATA"
echo "  Test Data:     $TEST_DATA"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  Run Name:      $RUN_NAME"
echo "============================================================"
echo ""

# Sanity checks
if [ ! -f "$TRAIN_DATA" ]; then
  echo "ERROR: train file not found: $TRAIN_DATA" >&2
  exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
  echo "ERROR: test file not found: $TEST_DATA" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting training..."
python experiments_downstream/models/train_predictor.py \
  --train-data "$TRAIN_DATA" \
  --test-data "$TEST_DATA" \
  --run-name "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo ""
echo "============================================================"
echo "Training Complete!"
echo "End: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
