#!/usr/bin/env python3
"""
Automated Ablation Study Pipeline (Option 3: Cross-Model Evaluation)

Runs the complete ablation study pipeline with parallel execution for training experiments.

Phases:
1. Generate synthetic data from 3 diffusion models (sequential)
2. Create hybrid datasets (sequential)
3. Train 9 cross-evaluation predictors (parallel)

Usage:
    python run_ablation_pipeline.py --benchmark ffmpeg
    python run_ablation_pipeline.py --benchmark iozone --skip-steps 1 2 3
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
    parser = argparse.ArgumentParser(description="Run ablation study pipeline (Option 3)")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g., ffmpeg, iozone, scimark2)")
    parser.add_argument("--scratch", default=os.environ.get("SCRATCH", "/scratch/yuvraj17"),
                       help="Scratch directory path")
    parser.add_argument("--checkpoint-epoch", type=int, default=19,
                       help="Checkpoint epoch to use (default: 19)")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="Number of synthetic samples to generate (default: 10000)")
    parser.add_argument("--skip-steps", nargs="+", type=int, default=[],
                       help="Steps to skip (e.g., --skip-steps 1 2 3)")
    parser.add_argument("--max-parallel", type=int, default=9,
                       help="Max parallel training jobs (default: 9)")
    
    args = parser.parse_args()
    
    # Configuration
    benchmark = args.benchmark
    scratch = args.scratch
    ckpt_epoch = args.checkpoint_epoch
    num_samples = args.num_samples
    
    # Paths
    exp_results = f"{scratch}/SyntheticLogGeneration/experiments_results"
    ablation_dir = f"{scratch}/SyntheticLogGeneration/experiments_downstream_results/ablation-diffusion/{benchmark}"
    real_data_dir = f"{scratch}/SyntheticLogGeneration/experiments_downstream_results/ablation/{benchmark}/data"
    
    print(f"\n{'='*80}")
    print(f"Ablation Study Pipeline (Option 3): {benchmark}")
    print(f"{'='*80}")
    print(f"Benchmark: {benchmark}")
    print(f"Num Samples: {num_samples}")
    print(f"Checkpoint Epoch: {ckpt_epoch}")
    print(f"Output Directory: {ablation_dir}")
    print(f"Skip Steps: {args.skip_steps if args.skip_steps else 'None'}")
    print(f"Max Parallel Jobs: {args.max_parallel}")
    print(f"{'='*80}\n")
    
    # Define sequential steps
    sequential_steps = {
        # Phase 1: Generate synthetic data
        1: {
            "name": "Generate Synthetic (Base Model)",
            "cmd": f'python sample_diffusion.py --ckpt "{exp_results}/exp_ablation_base_{benchmark}/ckpt_epoch_{ckpt_epoch}.pt" --out "{ablation_dir}/synthetic_base_10k.npz" --num-samples {num_samples} --seq-len 1024 --d-model 512 --nhead 8 --num-layers 8 --use-ddim --ddim-steps 50'
        },
        2: {
            "name": "Generate Synthetic (System Model)",
            "cmd": f'python sample_diffusion.py --ckpt "{exp_results}/exp_ablation_system_{benchmark}/ckpt_epoch_{ckpt_epoch}.pt" --out "{ablation_dir}/synthetic_system_10k.npz" --num-samples {num_samples} --seq-len 1024 --d-model 512 --nhead 8 --num-layers 8 --use-ddim --ddim-steps 50'
        },
        3: {
            "name": "Generate Synthetic (Full Model)",
            "cmd": f'python sample_diffusion.py --ckpt "{exp_results}/exp_ablation_full_{benchmark}/ckpt_epoch_{ckpt_epoch}.pt" --out "{ablation_dir}/synthetic_full_10k.npz" --num-samples {num_samples} --seq-len 1024 --d-model 512 --nhead 8 --num-layers 8 --use-ddim --ddim-steps 50'
        },
        # Phase 2: Create hybrid datasets
        4: {
            "name": "Create Hybrid (Base)",
            "cmd": f'python experiments_downstream/combine_datasets.py --real-data "{real_data_dir}/real_train.npz" --synthetic-data "{ablation_dir}/synthetic_base_10k.npz" --output "{ablation_dir}/hybrid_base_50_50.npz" --ratio 0.5'
        },
        5: {
            "name": "Create Hybrid (System)",
            "cmd": f'python experiments_downstream/combine_datasets.py --real-data "{real_data_dir}/real_train.npz" --synthetic-data "{ablation_dir}/synthetic_system_10k.npz" --output "{ablation_dir}/hybrid_system_50_50.npz" --ratio 0.5'
        },
        6: {
            "name": "Create Hybrid (Full)",
            "cmd": f'python experiments_downstream/combine_datasets.py --real-data "{real_data_dir}/real_train.npz" --synthetic-data "{ablation_dir}/synthetic_full_10k.npz" --output "{ablation_dir}/hybrid_full_50_50.npz" --ratio 0.5'
        },
    }
    
    # Define parallel training steps
    parallel_steps = {
        # Row 1: Base diffusion model
        7: {
            "name": "Train: Base → event",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_base_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event --run-name cross_base_event --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
        8: {
            "name": "Train: Base → event+dt",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_base_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event dt --run-name cross_base_base --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
        # Row 2: System diffusion model
        9: {
            "name": "Train: System → event",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_system_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event --run-name cross_system_event --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
        10: {
            "name": "Train: System → event+dt",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_system_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event dt --run-name cross_system_base --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
        11: {
            "name": "Train: System → event+dt+cpu+tid",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_system_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event dt cpu tid --run-name cross_system_system --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
        # Row 3: Full diffusion model
        12: {
            "name": "Train: Full → event",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_full_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event --run-name cross_full_event --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
        13: {
            "name": "Train: Full → event+dt",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_full_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event dt --run-name cross_full_base --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
        14: {
            "name": "Train: Full → event+dt+cpu+tid",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_full_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event dt cpu tid --run-name cross_full_system --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
        15: {
            "name": "Train: Full → all 6",
            "cmd": f'python experiments_downstream/models/train_predictor.py --train-data "{ablation_dir}/hybrid_full_50_50.npz" --test-data "{real_data_dir}/real_test.npz" --channels event dt cpu tid comm ret --run-name cross_full_full --output-dir "{ablation_dir}/cross-results" --seq-len 128 --batch-size 64 --epochs 20'
        },
    }
    
    # Run sequential steps (1-6)
    print("\n" + "="*80)
    print("PHASE 1 & 2: Sequential Steps (Generate + Combine)")
    print("="*80)
    
    for step_num in range(1, 7):
        if step_num in args.skip_steps:
            print(f"\n[Step {step_num}] ⏭️  Skipped")
            continue
        
        step = sequential_steps[step_num]
        success, _ = run_command(step["cmd"], f"Step {step_num}: {step['name']}")
        
        if not success:
            print(f"\n❌ Pipeline failed at Step {step_num}")
            sys.exit(1)
    
    # Run parallel training steps (7-15)
    print("\n" + "="*80)
    print("PHASE 3: Parallel Training (9 Cross-Evaluation Experiments)")
    print("="*80)
    print(f"Running up to {args.max_parallel} training experiments in parallel...\n")
    
    active_parallel_steps = {k: v for k, v in parallel_steps.items() if k not in args.skip_steps}
    
    if active_parallel_steps:
        with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = {
                executor.submit(run_command, step["cmd"], f"Step {num}: {step['name']}"): num
                for num, step in active_parallel_steps.items()
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
    
    print("\n" + "="*80)
    print("✅ Ablation Pipeline Completed Successfully!")
    print("="*80)
    print(f"\nResults saved to: {ablation_dir}/cross-results")
    print(f"\nRun analysis:")
    print(f"  python analyze_ablation_results.py --benchmark {benchmark}\n")


if __name__ == "__main__":
    main()
