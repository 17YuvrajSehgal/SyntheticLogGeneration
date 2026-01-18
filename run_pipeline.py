#!/usr/bin/env python3
"""
Automated Experiment Pipeline Runner

Runs the complete synthetic data generation and evaluation pipeline for any benchmark and window size.
Steps 6-9 (training experiments) run in parallel for efficiency.

Usage:
    python run_pipeline.py --benchmark scimark2 --window 256
    python run_pipeline.py --benchmark pybench --window 1024
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_command(cmd, step_name):
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    print(f"[{step_name}] Starting...")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(result.stdout)
        print(f"[{step_name}] ✅ Completed successfully")
        return True, step_name
    except subprocess.CalledProcessError as e:
        print(f"[{step_name}] ❌ Failed with error:")
        print(e.stdout)
        return False, step_name

def main():
    parser = argparse.ArgumentParser(description="Run complete experiment pipeline")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g., scimark2, pybench)")
    parser.add_argument("--window", type=int, required=True, help="Window size (256, 1024, 4096)")
    parser.add_argument("--scratch", default=os.environ.get("SCRATCH", "/scratch/yuvraj17"),
                       help="Scratch directory path")
    parser.add_argument("--checkpoint-epoch", type=int, default=19,
                       help="Checkpoint epoch to use (default: 19)")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="Number of synthetic samples to generate (default: 10000)")
    parser.add_argument("--skip-steps", nargs="+", type=int, default=[],
                       help="Steps to skip (e.g., --skip-steps 1 2 3)")
    
    args = parser.parse_args()
    
    # Configuration
    benchmark = args.benchmark
    window = args.window
    scratch = args.scratch
    ckpt_epoch = args.checkpoint_epoch
    num_samples = args.num_samples
    
    # Paths
    repo = f"{scratch}/SyntheticLogGeneration"
    data_dir = f"{scratch}/SyntheticLogGeneration/experiments_downstream_results/{benchmark}/{window}/data"
    results_dir = f"{scratch}/SyntheticLogGeneration/experiments_downstream_results/{benchmark}/{window}/results"
    ckpt_path = f"{scratch}/SyntheticLogGeneration/logs_tensorboard/improved_baseline_{benchmark}_{window}/ckpt_epoch_{ckpt_epoch}.pt"
    
    # Batch size based on window size
    if window <= 256:
        sample_batch = 64
        train_batch = 64
    elif window <= 1024:
        sample_batch = 32
        train_batch = 64
    else:
        sample_batch = 8
        train_batch = 32
    
    print(f"\n{'='*60}")
    print(f"Experiment Pipeline Configuration")
    print(f"{'='*60}")
    print(f"Benchmark: {benchmark}")
    print(f"Window Size: {window}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Num Samples: {num_samples}")
    print(f"Data Directory: {data_dir}")
    print(f"Results Directory: {results_dir}")
    print(f"Skip Steps: {args.skip_steps if args.skip_steps else 'None'}")
    print(f"{'='*60}\n")
    
    # Define all steps
    steps = {
        1: {
            "name": "Prepare Real Data",
            "cmd": f'python experiments_downstream/prepare_data.py --real-glob "{scratch}/window_shards/windowed_npz_{window}/{benchmark}/train/*.npz" --benchmark {benchmark} --output-dir "{data_dir}"'
        },
        2: {
            "name": "Generate Synthetic Data",
            "cmd": f'python sample_diffusion.py --ckpt "{ckpt_path}" --out "{data_dir}/synthetic_raw_{window}_{num_samples//1000}k.npz" --num-samples {num_samples} --seq-len {window} --batch-size {sample_batch} --d-model 256 --nhead 4 --num-layers 4 --use-ddim --ddim-steps 50'
        },
        3: {
            "name": "Create Combined (No Repair)",
            "cmd": f'python experiments_downstream/combine_datasets.py --real-data "{data_dir}/real_train.npz" --synthetic-data "{data_dir}/synthetic_raw_{window}_{num_samples//1000}k.npz" --output "{data_dir}/combined_real_synthetic_norepair_{window}_50_50.npz" --ratio 0.5'
        },
        4: {
            "name": "Repair Synthetic Data",
            "cmd": f'python synthetic_log_gen/repair.py --trace "{data_dir}/synthetic_raw_{window}_{num_samples//1000}k.npz" --constraints dataset/constraints_universal.json --output "{data_dir}/synthetic_repaired_{window}_{num_samples//1000}k.npz"'
        },
        5: {
            "name": "Create Combined (Repaired)",
            "cmd": f'python experiments_downstream/combine_datasets.py --real-data "{data_dir}/real_train.npz" --synthetic-data "{data_dir}/synthetic_repaired_{window}_{num_samples//1000}k.npz" --output "{data_dir}/combined_real_synthetic_repaired_{window}_50_50.npz" --ratio 0.5'
        },
        6: {
            "name": "Train on Real Data (Baseline)",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{data_dir}/real_train.npz" --test-data "{data_dir}/real_test.npz" --run-name "real_baseline_{benchmark}_{window}" --output-dir "{results_dir}" --seq-len 128 --batch-size {train_batch} --epochs 20'
        },
        7: {
            "name": "Train on Synthetic Only",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{data_dir}/synthetic_raw_{window}_{num_samples//1000}k.npz" --test-data "{data_dir}/real_test.npz" --run-name "synthetic_data_only_{benchmark}_{window}" --output-dir "{results_dir}" --seq-len 128 --batch-size {train_batch} --epochs 20'
        },
        8: {
            "name": "Train on Combined (Repaired)",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{data_dir}/combined_real_synthetic_repaired_{window}_50_50.npz" --test-data "{data_dir}/real_test.npz" --run-name "combined_50_50_{benchmark}_{window}" --output-dir "{results_dir}" --seq-len 128 --batch-size {train_batch} --epochs 20'
        },
        9: {
            "name": "Train on Combined (No Repair)",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{data_dir}/combined_real_synthetic_norepair_{window}_50_50.npz" --test-data "{data_dir}/real_test.npz" --run-name "combined_50_50_norepair_{benchmark}_{window}" --output-dir "{results_dir}" --seq-len 128 --batch-size {train_batch} --epochs 20'
        }
    }
    
    # Run sequential steps (1-5)
    print("\n" + "="*60)
    print("PHASE 1: Sequential Steps (1-5)")
    print("="*60)
    
    for step_num in range(1, 6):
        if step_num in args.skip_steps:
            print(f"\n[Step {step_num}] ⏭️  Skipped")
            continue
        
        step = steps[step_num]
        success, _ = run_command(step["cmd"], f"Step {step_num}: {step['name']}")
        
        if not success:
            print(f"\n❌ Pipeline failed at Step {step_num}")
            sys.exit(1)
    
    # Run parallel steps (6-9)
    print("\n" + "="*60)
    print("PHASE 2: Parallel Steps (6-9) - Training Experiments")
    print("="*60)
    print("Running 4 training experiments in parallel...\n")
    
    parallel_steps = {k: v for k, v in steps.items() if k >= 6 and k not in args.skip_steps}
    
    if parallel_steps:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_command, step["cmd"], f"Step {num}: {step['name']}"): num
                for num, step in parallel_steps.items()
            }
            
            completed = 0
            total = len(futures)
            failed_steps = []
            
            for future in as_completed(futures):
                step_num = futures[future]
                success, step_name = future.result()
                completed += 1
                
                if not success:
                    failed_steps.append(step_num)
                
                print(f"\n[Progress] {completed}/{total} training experiments completed")
            
            if failed_steps:
                print(f"\n❌ Some training experiments failed: Steps {failed_steps}")
                sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Pipeline Completed Successfully!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print(f"Data saved to: {data_dir}\n")

if __name__ == "__main__":
    main()
