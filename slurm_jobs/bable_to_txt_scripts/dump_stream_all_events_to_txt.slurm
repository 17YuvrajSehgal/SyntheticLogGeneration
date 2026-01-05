#!/bin/bash
#SBATCH --job-name=bt2_dump_stream
#SBATCH --account=def-naser2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/stream
#_dump_%j.out
#SBATCH --error=logs//stream
#_dump_%j.err

set -euo pipefail

echo "===== Babeltrace2 dump job ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Time: $(date)"
echo "================================"

# Adjust if your path differs
BASE="/home/yuvraj17/scratch/unzipped-benchmarks/stream/stream"
OUTDIR="${BASE}/txt_traces"

mkdir -p "${OUTDIR}" logs

# (Optional) load modules if your cluster uses module system
# module purge
# module load babeltrace2 2>/dev/null || true

# If babeltrace2 isn't found, fail fast with a helpful message
command -v babeltrace2 >/dev/null 2>&1 || {
  echo "ERROR: babeltrace2 not found in PATH. Load the module or install it."
  exit 1
}

echo "Input base: ${BASE}"
echo "Output dir: ${OUTDIR}"
echo

# Iterate over all all-events runs
shopt -s nullglob
count=0

for run_kernel in "${BASE}"/stream-all-events-run*/kernel; do
  run_parent="$(dirname "${run_kernel}")"          
  run_name="$(basename "${run_parent}")"            
  out_file="${OUTDIR}/${run_name}.txt"

  echo "Dumping ${run_name} -> ${out_file}"
  # Use srun so Slurm accounts for CPU usage properly
  srun --exclusive -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" \
    babeltrace2 "${run_kernel}" > "${out_file}"

  # quick sanity check
  if [[ ! -s "${out_file}" ]]; then
    echo "WARNING: ${out_file} is empty!"
  fi

  count=$((count + 1))
done

echo
echo "Done. Wrote ${count} txt files into ${OUTDIR}"
echo "Finished at: $(date)"
