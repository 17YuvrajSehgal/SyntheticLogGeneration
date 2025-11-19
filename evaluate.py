"""
Evaluation script for generated kernel traces.
Implements metrics from the ICSE 2025 paper: accuracy, perfect rate, and ROUGE-L.
"""

import numpy as np
import argparse
from collections import Counter
import os


def longest_common_subsequence(seq1, seq2):
    """Compute the longest common subsequence (LCS) between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def accuracy(original, reconstructed):
    """
    Calculate accuracy: proportion of events correctly positioned.
    
    Args:
        original: Original sequence
        reconstructed: Reconstructed sequence
    
    Returns:
        Accuracy score (0-1)
    """
    if len(original) != len(reconstructed):
        return 0.0
    
    correct = sum(1 for o, r in zip(original, reconstructed) if o == r)
    return correct / len(original)


def perfect_rate(original_sequences, reconstructed_sequences):
    """
    Calculate perfect rate: proportion of sequences perfectly reconstructed.
    
    Args:
        original_sequences: List of original sequences
        reconstructed_sequences: List of reconstructed sequences
    
    Returns:
        Perfect rate (0-1)
    """
    if len(original_sequences) != len(reconstructed_sequences):
        return 0.0
    
    perfect = 0
    for orig, recon in zip(original_sequences, reconstructed_sequences):
        if len(orig) == len(recon) and all(o == r for o, r in zip(orig, recon)):
            perfect += 1
    
    return perfect / len(original_sequences)


def rouge_l(original, reconstructed):
    """
    Calculate ROUGE-L score based on LCS.
    
    Args:
        original: Original sequence
        reconstructed: Reconstructed sequence
    
    Returns:
        ROUGE-L score (0-1)
    """
    if len(original) == 0:
        return 1.0 if len(reconstructed) == 0 else 0.0
    
    lcs = longest_common_subsequence(original, reconstructed)
    return lcs / len(original)


def evaluate_against_real(generated_sequences, real_sequences):
    """
    Evaluate generated sequences against real sequences.
    This finds the closest real sequence for each generated sequence.
    
    Args:
        generated_sequences: List of generated sequences
        real_sequences: List of real sequences
    
    Returns:
        Dictionary with evaluation metrics
    """
    accuracies = []
    rouge_scores = []
    
    for gen_seq in generated_sequences:
        best_acc = 0.0
        best_rouge = 0.0
        
        # Find closest real sequence
        for real_seq in real_sequences:
            acc = accuracy(real_seq, gen_seq)
            rouge = rouge_l(real_seq, gen_seq)
            
            if acc > best_acc:
                best_acc = acc
            if rouge > best_rouge:
                best_rouge = rouge
        
        accuracies.append(best_acc)
        rouge_scores.append(best_rouge)
    
    return {
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'rouge_l': np.mean(rouge_scores),
        'rouge_l_std': np.std(rouge_scores)
    }


def evaluate_distribution(generated_sequences, real_sequences):
    """
    Evaluate the distribution of generated sequences.
    
    Args:
        generated_sequences: List of generated sequences
        real_sequences: List of real sequences
    
    Returns:
        Dictionary with distribution metrics
    """
    # Flatten sequences
    gen_flat = [item for seq in generated_sequences for item in seq]
    real_flat = [item for seq in real_sequences for item in seq]
    
    # Count frequencies
    gen_counts = Counter(gen_flat)
    real_counts = Counter(real_flat)
    
    # Get all unique events
    all_events = set(gen_flat) | set(real_flat)
    
    # Compute KL divergence (simplified)
    gen_total = len(gen_flat)
    real_total = len(real_flat)
    
    kl_div = 0.0
    for event in all_events:
        gen_prob = gen_counts.get(event, 0) / gen_total
        real_prob = real_counts.get(event, 0) / real_total
        
        if real_prob > 0 and gen_prob > 0:
            kl_div += real_prob * np.log(real_prob / gen_prob)
    
    return {
        'kl_divergence': kl_div,
        'gen_unique_events': len(set(gen_flat)),
        'real_unique_events': len(set(real_flat)),
        'gen_mean_length': np.mean([len(seq) for seq in generated_sequences]),
        'real_mean_length': np.mean([len(seq) for seq in real_sequences])
    }


def load_sequences(file_path):
    """Load sequences from a text file."""
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            seq = [int(x) for x in line.strip().split(',')]
            sequences.append(seq)
    return sequences


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated kernel traces')
    parser.add_argument('--generated', type=str, required=True,
                       help='Path to generated traces file')
    parser.add_argument('--real', type=str, required=True,
                       help='Path to real traces file for comparison')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation results (JSON)')
    
    args = parser.parse_args()
    
    # Load sequences
    print(f"Loading generated sequences from {args.generated}")
    generated_sequences = load_sequences(args.generated)
    
    print(f"Loading real sequences from {args.real}")
    real_sequences = load_sequences(args.real)
    
    print(f"Loaded {len(generated_sequences)} generated sequences")
    print(f"Loaded {len(real_sequences)} real sequences")
    
    # Evaluate
    print("Evaluating...")
    
    # Evaluation against real sequences
    eval_results = evaluate_against_real(generated_sequences, real_sequences)
    
    # Distribution evaluation
    dist_results = evaluate_distribution(generated_sequences, real_sequences)
    
    # Combine results
    results = {
        'similarity_metrics': eval_results,
        'distribution_metrics': dist_results
    }
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {eval_results['accuracy']:.4f} ± {eval_results['accuracy_std']:.4f}")
    print(f"ROUGE-L: {eval_results['rouge_l']:.4f} ± {eval_results['rouge_l_std']:.4f}")
    print(f"\nDistribution Metrics:")
    print(f"KL Divergence: {dist_results['kl_divergence']:.4f}")
    print(f"Generated unique events: {dist_results['gen_unique_events']}")
    print(f"Real unique events: {dist_results['real_unique_events']}")
    print(f"Generated mean length: {dist_results['gen_mean_length']:.2f}")
    print(f"Real mean length: {dist_results['real_mean_length']:.2f}")
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

