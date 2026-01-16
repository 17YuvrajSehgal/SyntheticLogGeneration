#!/usr/bin/env python3
"""
Generate Slurm job scripts for running pipelines on different benchmarks and window sizes.

Usage:
    python generate_pipeline_jobs.py
"""

import os


def create_slurm_job(benchmark, window, output_dir="slurm_jobs/pipelines"):
    """Create a Slurm job script for a specific benchmark and window size."""
    
    # Read template
    with open("slurm_jobs/pipeline_template.slurm", "r") as f:
        template = f.read()
    
    # Replace placeholders
    content = template.replace("{BENCHMARK}", benchmark).replace("{WINDOW}", str(window))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write job script
    output_file = f"{output_dir}/run_{benchmark}_{window}.slurm"
    with open(output_file, "w") as f:
        f.write(content)
    
    print(f"Created: {output_file}")
    return output_file

def main():
    # Define benchmarks and window sizes
    benchmarks = ["scimark2", "pybench", "ffmpeg", "stream"]
    windows = [256, 1024, 4096]
    
    print("="*60)
    print("Generating Pipeline Slurm Jobs")
    print("="*60)
    
    created_files = []
    
    for benchmark in benchmarks:
        for window in windows:
            job_file = create_slurm_job(benchmark, window)
            created_files.append(job_file)
    
    print(f"\n{'='*60}")
    print(f"✅ Created {len(created_files)} Slurm job scripts")
    print(f"{'='*60}\n")
    
    print("To submit jobs:")
    print("  cd slurm_jobs/pipelines")
    print("  sbatch run_scimark2_256.slurm")
    print("  sbatch run_pybench_1024.slurm")
    print("  # etc...\n")
    
    print("Or submit all at once:")
    print("  for f in slurm_jobs/pipelines/*.slurm; do sbatch $f; done\n")

if __name__ == "__main__":
    main()
