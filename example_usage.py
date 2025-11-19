"""
Example usage script demonstrating how to use the synthetic trace generation pipeline.
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print("✓ Success!")
    else:
        print(f"✗ Error: {result.stderr}")
        return False
    
    return True


def main():
    """Run example pipeline."""
    
    # Configuration
    dataset = "compress-gzip"
    sequence_length = 200
    epochs = 10  # Reduced for quick example
    
    print("="*60)
    print("Synthetic Kernel Trace Generation - Example Usage")
    print("="*60)
    
    # Step 1: Analyze dataset
    data_path = f"Datasets/{dataset}/sequence_length_{sequence_length}/training"
    if os.path.exists(data_path):
        run_command(
            f"python utils/data_stats.py --data_path {data_path}",
            "Step 1: Analyzing dataset statistics"
        )
    else:
        print(f"Warning: Dataset not found at {data_path}")
        print("Please ensure the dataset exists before running this example.")
        return
    
    # Step 2: Train model
    checkpoint_dir = f"checkpoints/{dataset}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_cmd = (
        f"python train.py "
        f"--dataset {dataset} "
        f"--sequence_length {sequence_length} "
        f"--batch_size 16 "
        f"--epochs {epochs} "
        f"--lr 2e-4 "
        f"--d_model 64 "
        f"--num_layers 2 "
        f"--save_dir {checkpoint_dir} "
        f"--device cuda"
    )
    
    print("\n" + "="*60)
    print("Step 2: Training the model")
    print("="*60)
    print("This will take some time. You can skip this step if you already have a trained model.")
    response = input("Do you want to train the model? (y/n): ")
    
    if response.lower() == 'y':
        run_command(train_cmd, "Training SSSDS4 model")
    else:
        print("Skipping training. Make sure you have a trained model checkpoint.")
    
    # Step 3: Generate traces
    checkpoint_path = f"{checkpoint_dir}/best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"{checkpoint_dir}/latest.pt"
    
    if os.path.exists(checkpoint_path):
        output_dir = f"generated_traces/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        
        generate_cmd = (
            f"python generate.py "
            f"--checkpoint {checkpoint_path} "
            f"--num_samples 50 "
            f"--seq_len {sequence_length} "
            f"--batch_size 8 "
            f"--output_dir {output_dir} "
            f"--output_format both"
        )
        
        run_command(generate_cmd, "Step 3: Generating synthetic traces")
        
        # Step 4: Evaluate
        generated_path = f"{output_dir}/generated_traces.txt"
        test_path = f"Datasets/{dataset}/sequence_length_{sequence_length}/testing"
        
        if os.path.exists(generated_path) and os.path.exists(test_path):
            eval_cmd = (
                f"python evaluate.py "
                f"--generated {generated_path} "
                f"--real {test_path} "
                f"--output {output_dir}/evaluation_results.json"
            )
            
            run_command(eval_cmd, "Step 4: Evaluating generated traces")
        else:
            print(f"Warning: Cannot evaluate - missing files")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")
    
    print("\n" + "="*60)
    print("Example pipeline completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated traces in the output directory")
    print("2. Check evaluation results")
    print("3. Adjust hyperparameters and retrain if needed")
    print("4. Experiment with different datasets")


if __name__ == "__main__":
    main()

