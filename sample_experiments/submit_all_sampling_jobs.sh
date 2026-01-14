#!/bin/bash
# Submit all sampling jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# Context 256
sbatch "$SCRIPT_DIR/sample_context_256_compress-gzip.slurm"
sbatch "$SCRIPT_DIR/sample_context_256_ffmpeg.slurm"
sbatch "$SCRIPT_DIR/sample_context_256_iozone.slurm"
sbatch "$SCRIPT_DIR/sample_context_256_phpbench.slurm"
sbatch "$SCRIPT_DIR/sample_context_256_pybench.slurm"
sbatch "$SCRIPT_DIR/sample_context_256_ramspeed.slurm"
sbatch "$SCRIPT_DIR/sample_context_256_scimark2.slurm"
sbatch "$SCRIPT_DIR/sample_context_256_stream.slurm"
sbatch "$SCRIPT_DIR/sample_context_256_unpack-linux.slurm"

# Context 1024
sbatch "$SCRIPT_DIR/sample_context_1024_compress-gzip.slurm"
sbatch "$SCRIPT_DIR/sample_context_1024_ffmpeg.slurm"
sbatch "$SCRIPT_DIR/sample_context_1024_iozone.slurm"
sbatch "$SCRIPT_DIR/sample_context_1024_phpbench.slurm"
sbatch "$SCRIPT_DIR/sample_context_1024_pybench.slurm"
sbatch "$SCRIPT_DIR/sample_context_1024_ramspeed.slurm"
sbatch "$SCRIPT_DIR/sample_context_1024_scimark2.slurm"
sbatch "$SCRIPT_DIR/sample_context_1024_stream.slurm"
sbatch "$SCRIPT_DIR/sample_context_1024_unpack-linux.slurm"

# Context 4096
sbatch "$SCRIPT_DIR/sample_context_4096_compress-gzip.slurm"
sbatch "$SCRIPT_DIR/sample_context_4096_ffmpeg.slurm"
sbatch "$SCRIPT_DIR/sample_context_4096_iozone.slurm"
sbatch "$SCRIPT_DIR/sample_context_4096_phpbench.slurm"
sbatch "$SCRIPT_DIR/sample_context_4096_pybench.slurm"
sbatch "$SCRIPT_DIR/sample_context_4096_ramspeed.slurm"
sbatch "$SCRIPT_DIR/sample_context_4096_scimark2.slurm"
sbatch "$SCRIPT_DIR/sample_context_4096_stream.slurm"
sbatch "$SCRIPT_DIR/sample_context_4096_unpack-linux.slurm"

echo '[INFO] All jobs submitted!'
