#!/bin/bash
#SBATCH --job-name=repair-synthetic
#SBATCH --account=def-naser2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ---- Environment ----
module --force purge
module load StdEnv/2023 python/3.11

source /project/6083487/yuvraj/SyntheticLogGeneration/.venv/bin/activate
cd /project/6083487/yuvraj/SyntheticLogGeneration

mkdir -p logs outputs

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"

# ---- Repair Synthetic Data ----
# TODO: Update --synth_npz to point to your actual generated file
# TODO: Update real_glob if using a different benchmark

SYNTH_INPUT="outputs/diffusion_outputs/generated_traces/compress-gzip/synth_100k_step50k.npz"
SYNTH_OUTPUT="generated_traces/discrete/compress-gzip/synth_100k_step50k_repaired.npz"
CONSTRAINTS="dataset/constraints.json"
REAL_GLOB="window_shards/compress-gzip/train/*.npz"

if [ ! -f "$CONSTRAINTS" ]; then
    echo "[ERROR] Constraints file not found: $CONSTRAINTS"
    echo "Please run 'sbatch slurm_jobs/run_learn_constraints.slurm' first."
    exit 1
fi

if [ ! -f "$SYNTH_INPUT" ]; then
    echo "[WARN] Input file not found: $SYNTH_INPUT"
    echo "Make sure you have generated traces first."
    # We exit 1, unless you want to debug with a dummy file
    exit 1
fi

python -u -m synthetic_data_processing.repair_synthetic \
  --synth_npz "$SYNTH_INPUT" \
  --constraints_json "$CONSTRAINTS" \
  --real_glob "$REAL_GLOB" \
  --output_npz "$SYNTH_OUTPUT" \
  --max_ref 10000

echo "Repair completed."
