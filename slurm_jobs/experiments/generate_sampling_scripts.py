#!/usr/bin/env python3
"""
Generate Slurm scripts for sampling synthetic traces from trained diffusion models.

This script creates Slurm job files for all benchmarks and context lengths.
The generated scripts automatically find the checkpoint with the highest epoch number.
"""

import os
from pathlib import Path

# Configuration
BENCHMARKS = [
    "compress-gzip",
    "ffmpeg", 
    "iozone",
    "phpbench",
    "pybench",
    "ramspeed",
    "scimark2",
    "stream",
    "unpack-linux"
]

CONTEXTS = {
    "256": {
        "d_model": 512,
        "nhead": 8,
        "num_layers": 8,
        "seq_len": 256,
        "num_samples": 1000,
        "batch_size": 32,
        "time": "04:00:00"
    },
    "1024": {
        "d_model": 512,
        "nhead": 8,
        "num_layers": 8,
        "seq_len": 1024,
        "num_samples": 1000,
        "batch_size": 32,
        "time": "04:00:00"
    },
    "4096": {
        "d_model": 512,
        "nhead": 8,
        "num_layers": 8,
        "seq_len": 4096,
        "num_samples": 1000,
        "batch_size": 32,
        "time": "04:00:00"
    }
}

# Paths
REPO_PATH = "/project/def-naser2/yuvraj/SyntheticLogGeneration"
OUTPUT_DIR = "sample_experiments"

# Template for Slurm script with automatic checkpoint detection
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=sample_{context}_{benchmark}
#SBATCH --account=def-naser2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={time}
#SBATCH --output={repo}/experiments_results/{benchmark}/exp_context_{context}/sample_%j.out
#SBATCH --error={repo}/experiments_results/{benchmark}/exp_context_{context}/sample_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

REPO="{repo}"
cd "$REPO"
source "$REPO/.venv/bin/activate"

echo "[START] Generating samples for {benchmark} (context={context})"
echo "Date: $(date)"

# Find the latest checkpoint (highest epoch number)
CKPT_DIR="experiments_results/{benchmark}/exp_context_{context}/exp_context_{context}_{benchmark}"
OUT="generated_traces/context_{context}/{benchmark}_samples.npz"

# Create output directory
mkdir -p "generated_traces/context_{context}"

# Check if checkpoint directory exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "[ERROR] Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

# Find all checkpoint files and extract epoch numbers
echo "[INFO] Searching for checkpoints in: $CKPT_DIR"
CKPT_FILES=("$CKPT_DIR"/ckpt_epoch_*.pt)

if [ ! -e "${{CKPT_FILES[0]}}" ]; then
    echo "[ERROR] No checkpoint files found in $CKPT_DIR"
    exit 1
fi

# Find checkpoint with highest epoch number
MAX_EPOCH=-1
BEST_CKPT=""

for ckpt_file in "${{CKPT_FILES[@]}}"; do
    # Extract epoch number from filename (e.g., ckpt_epoch_19.pt -> 19)
    filename=$(basename "$ckpt_file")
    epoch_num=$(echo "$filename" | sed -n 's/ckpt_epoch_\\([0-9]*\\)\\.pt/\\1/p')
    
    if [ -n "$epoch_num" ] && [ "$epoch_num" -gt "$MAX_EPOCH" ]; then
        MAX_EPOCH=$epoch_num
        BEST_CKPT="$ckpt_file"
    fi
done

if [ -z "$BEST_CKPT" ]; then
    echo "[ERROR] Could not find valid checkpoint files"
    exit 1
fi

CKPT="$BEST_CKPT"
echo "[INFO] Found ${{#CKPT_FILES[@]}} checkpoint(s)"
echo "[INFO] Using checkpoint with highest epoch: $CKPT (epoch $MAX_EPOCH)"
echo "[INFO] Output: $OUT"

python -u sample_diffusion.py \\
    --ckpt "$CKPT" \\
    --out "$OUT" \\
    --d-model {d_model} \\
    --nhead {nhead} \\
    --num-layers {num_layers} \\
    --steps 1000 \\
    --seq-len {seq_len} \\
    --num-samples {num_samples} \\
    --batch-size {batch_size} \\
    --vocab-dir dataset/metadata_all_events \\
    --num-cpus 4 \\
    --tid-buckets 256 \\
    --fd-cap 1025 \\
    --device cuda

echo "[DONE] Saved to $OUT"
echo "Date: $(date)"
"""

def generate_slurm_scripts():
    """Generate Slurm scripts for all benchmarks and context lengths."""
    
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    for context, config in CONTEXTS.items():
        for benchmark in BENCHMARKS:
            # Generate script content
            script_content = SLURM_TEMPLATE.format(
                context=context,
                benchmark=benchmark,
                repo=REPO_PATH,
                time=config["time"],
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                seq_len=config["seq_len"],
                num_samples=config["num_samples"],
                batch_size=config["batch_size"]
            )
            
            # Write to file
            filename = f"sample_context_{context}_{benchmark}.slurm"
            filepath = output_path / filename
            
            with open(filepath, 'w', newline='\n') as f:
                f.write(script_content)
            
            generated_files.append(str(filepath))
            print(f"✓ Generated: {filepath}")
    
    return generated_files

def generate_submit_all_script():
    """Generate a convenience script to submit all jobs."""
    
    output_path = Path(OUTPUT_DIR)
    submit_script = output_path / "submit_all_sampling_jobs.sh"
    
    with open(submit_script, 'w', newline='\n') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all sampling jobs\n\n")
        f.write("set -euo pipefail\n\n")
        f.write("SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\n")
        
        for context in CONTEXTS.keys():
            f.write(f"\n# Context {context}\n")
            for benchmark in BENCHMARKS:
                filename = f"sample_context_{context}_{benchmark}.slurm"
                f.write(f"sbatch \"$SCRIPT_DIR/{filename}\"\n")
        
        f.write("\necho '[INFO] All jobs submitted!'\n")
    
    # Make executable
    os.chmod(submit_script, 0o755)
    print(f"\n✓ Generated submission script: {submit_script}")
    return str(submit_script)

def main():
    print("=" * 60)
    print("Generating Slurm Scripts for Sampling Synthetic Traces")
    print("=" * 60)
    print(f"\nBenchmarks: {len(BENCHMARKS)}")
    print(f"Contexts: {list(CONTEXTS.keys())}")
    print(f"Total scripts: {len(BENCHMARKS) * len(CONTEXTS)}")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Generate individual scripts
    generated_files = generate_slurm_scripts()
    
    # Generate submit-all script
    submit_script = generate_submit_all_script()
    
    print(f"\n{'=' * 60}")
    print(f"✓ Successfully generated {len(generated_files)} Slurm scripts")
    print(f"✓ Submission script: {submit_script}")
    print(f"{'=' * 60}")
    print("\nKey Features:")
    print("  • Automatically finds checkpoint with highest epoch number")
    print("  • Handles missing checkpoints gracefully")
    print("  • Optimized batch sizes for each context length")
    print("\nTo submit all jobs:")
    print(f"  cd {OUTPUT_DIR}")
    print(f"  ./submit_all_sampling_jobs.sh")
    print("\nOr submit individual jobs:")
    print(f"  sbatch {OUTPUT_DIR}/sample_context_256_scimark2.slurm")

if __name__ == "__main__":
    main()
