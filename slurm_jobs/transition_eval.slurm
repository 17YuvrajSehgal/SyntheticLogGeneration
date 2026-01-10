#!/bin/bash
#SBATCH --job-name=transition-eval
#SBATCH --account=def-naser2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ---- Environment ----
module --force purge
module load StdEnv/2023 python/3.11

source /project/6083487/yuvraj/SyntheticLogGeneration/.venv/bin/activate
cd /project/6083487/yuvraj/SyntheticLogGeneration

mkdir -p logs

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "Python: $(which python)"
python -c "import numpy; print('NumPy:', numpy.__version__)"

# ---- Transition Evaluation ----
python -u dataset_processing/transition_eval.py \
  --real_glob "dataset/window_shards/compress-gzip/train/*.npz" \
  --synth generated_traces/discrete/compress-gzip/synth_100k_step50k_repaired.npz \
  --max_real_shards 50 \
  --num_events 384 \
  --num_dt_buckets 256 \
  --num_cpus 4 \
  --topk_events_dt 30 \
  --min_occ_dt 5000

echo "Transition evaluation completed."

#original sample
# python -u evaluation_metrics/transition_eval.py --real_glob "dataset/window_shards/compress-gzip/train/*.npz" --synth generated_traces/discrete/compress-gzip/synth_100k_step50k_repaired.npz --max_real_shards 50 --num_events 384 --num_dt_buckets 256 --num_cpus 4 --topk_events_dt 30 --min_occ_dt 5000

#repaired sample
# python -u evaluation_metrics/transition_eval.py --real_glob "dataset/window_shards/compress-gzip/train/*.npz" --synth generated_traces/discrete/compress-gzip/synth_100k_step50k_repaired.npz --max_real_shards 50 --num_events 384 --num_dt_buckets 256 --num_cpus 4 --topk_events_dt 30 --min_occ_dt 5000