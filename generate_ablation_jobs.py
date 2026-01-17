#!/usr/bin/env python3
"""
Generate Slurm job scripts for ablation study on different benchmarks.

Usage:
    python generate_ablation_jobs.py
"""

import os


def create_slurm_job(benchmark, output_dir="slurm_jobs/ablation"):
    """Create a Slurm job script for a specific benchmark."""
    
    # Read template
    with open("slurm_jobs/ablation_template.slurm", "r") as f:
        template = f.read()
    
    # Replace placeholders
    content = template.replace("{BENCHMARK}", benchmark)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write job script
    output_file = f"{output_dir}/run_ablation_{benchmark}.slurm"
    with open(output_file, "w") as f:
        f.write(content)
    
    print(f"Created: {output_file}")
    return output_file


def main():
    # Define benchmarks for ablation study
    benchmarks = ["ffmpeg", "pybench", "scimark2"]
    
    print("="*60)
    print("Generating Ablation Study Slurm Jobs")
    print("="*60)
    
    created_files = []
    
    for benchmark in benchmarks:
        job_file = create_slurm_job(benchmark)
        created_files.append(job_file)
    
    print(f"\n{'='*60}")
    print(f"✅ Created {len(created_files)} Slurm job scripts")
    print(f"{'='*60}\n")
    
    print("To submit jobs:")
    print("  cd slurm_jobs/ablation")
    print("  sbatch run_ablation_ffmpeg.slurm")
    print("  sbatch run_ablation_iozone.slurm")
    print("  sbatch run_ablation_scimark2.slurm\n")
    
    print("Or submit all at once:")
    print("  for f in slurm_jobs/ablation/*.slurm; do sbatch $f; done\n")
    
    print("To run locally (for testing):")
    print("  python run_ablation_pipeline.py --benchmark ffmpeg\n")


if __name__ == "__main__":
    main()
