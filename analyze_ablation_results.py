#!/usr/bin/env python3
"""
Collect and analyze ablation study results.

This script collects F1 scores from all cross-evaluation experiments
and generates a comprehensive results table.

Usage:
    python analyze_ablation_results.py --benchmark ffmpeg
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd


def load_metrics(result_dir):
    """Load final_metrics.json from a result directory."""
    metrics_file = os.path.join(result_dir, 'final_metrics.json')
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)


def collect_cross_results(base_dir):
    """Collect all cross-evaluation results."""
    results = []
    
    # Define the matrix
    experiments = [
        # (diffusion_model, predictor_channels, run_name)
        ('Base', 'event', 'cross_base_event'),
        ('Base', 'event+dt', 'cross_base_base'),
        
        ('System', 'event', 'cross_system_event'),
        ('System', 'event+dt', 'cross_system_base'),
        ('System', 'event+dt+cpu+tid', 'cross_system_system'),
        
        ('Full', 'event', 'cross_full_event'),
        ('Full', 'event+dt', 'cross_full_base'),
        ('Full', 'event+dt+cpu+tid', 'cross_full_system'),
        ('Full', 'all 6', 'cross_full_full'),
    ]
    
    for diffusion, predictor, run_name in experiments:
        result_dir = os.path.join(base_dir, run_name)
        metrics = load_metrics(result_dir)
        
        if metrics:
            results.append({
                'Diffusion Model': diffusion,
                'Predictor Channels': predictor,
                'F1 (Macro)': metrics.get('f1_macro', 0.0),
                'Accuracy': metrics.get('accuracy', 0.0),
                'F1 (Weighted)': metrics.get('f1_weighted', 0.0),
                'Top-5 Acc': metrics.get('top5_accuracy', 0.0),
            })
        else:
            results.append({
                'Diffusion Model': diffusion,
                'Predictor Channels': predictor,
                'F1 (Macro)': None,
                'Accuracy': None,
                'F1 (Weighted)': None,
                'Top-5 Acc': None,
            })
    
    return pd.DataFrame(results)


def create_matrix_table(df):
    """Create a pivot table for the cross-evaluation matrix."""
    # Create pivot table
    pivot = df.pivot(
        index='Diffusion Model',
        columns='Predictor Channels',
        values='F1 (Macro)'
    )
    
    # Reorder columns
    col_order = ['event', 'event+dt', 'event+dt+cpu+tid', 'all 6']
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    
    # Reorder rows
    row_order = ['Base', 'System', 'Full']
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    
    return pivot


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('--benchmark', required=True, help='Benchmark name (e.g., ffmpeg)')
    parser.add_argument('--base-dir', default='$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation-diffusion',
                       help='Base directory for results')
    
    args = parser.parse_args()
    
    # Expand environment variables
    base_dir = os.path.expandvars(args.base_dir)
    results_dir = os.path.join(base_dir, args.benchmark, 'cross-results')
    
    print(f"{'='*80}")
    print(f"Ablation Study Results: {args.benchmark}")
    print(f"{'='*80}\n")
    
    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run the experiments first!")
        return
    
    # Collect results
    print("Collecting results...")
    df = collect_cross_results(results_dir)
    
    # Print detailed results
    print("\n" + "="*80)
    print("Detailed Results")
    print("="*80)
    print(df.to_string(index=False))
    
    # Create matrix table
    print("\n" + "="*80)
    print("Cross-Evaluation Matrix (F1 Macro)")
    print("="*80)
    matrix = create_matrix_table(df)
    
    # Format as percentage
    matrix_pct = matrix * 100
    print(matrix_pct.to_string(float_format=lambda x: f'{x:.2f}%' if pd.notna(x) else '-'))
    
    # Calculate improvements
    print("\n" + "="*80)
    print("Key Findings")
    print("="*80)
    
    # Diagonal values (optimal configurations)
    diagonal = []
    if 'event+dt' in matrix.columns and 'Base' in matrix.index:
        diagonal.append(('Base → event+dt', matrix.loc['Base', 'event+dt']))
    if 'event+dt+cpu+tid' in matrix.columns and 'System' in matrix.index:
        diagonal.append(('System → event+dt+cpu+tid', matrix.loc['System', 'event+dt+cpu+tid']))
    if 'all 6' in matrix.columns and 'Full' in matrix.index:
        diagonal.append(('Full → all 6', matrix.loc['Full', 'all 6']))
    
    print("\n1. Diagonal (Optimal Configurations):")
    for name, value in diagonal:
        if pd.notna(value):
            print(f"   {name}: {value*100:.2f}%")
    
    # Row improvements (same predictor, better diffusion)
    if 'all 6' in matrix.columns:
        print("\n2. Diffusion Model Quality (using all 6 channels in predictor):")
        for model in ['Base', 'System', 'Full']:
            if model in matrix.index:
                value = matrix.loc[model, 'all 6']
                if pd.notna(value):
                    print(f"   {model}: {value*100:.2f}%")
    
    # Column improvements (same diffusion, better predictor)
    if 'Full' in matrix.index:
        print("\n3. Predictor Channel Importance (using Full diffusion model):")
        for channels in matrix.columns:
            value = matrix.loc['Full', channels]
            if pd.notna(value):
                print(f"   {channels}: {value*100:.2f}%")
    
    # Save to CSV
    output_file = os.path.join(results_dir, f'ablation_results_{args.benchmark}.csv')
    df.to_csv(output_file, index=False)
    print(f"\n✅ Detailed results saved to: {output_file}")
    
    # Save matrix
    matrix_file = os.path.join(results_dir, f'ablation_matrix_{args.benchmark}.csv')
    matrix.to_csv(matrix_file)
    print(f"✅ Matrix saved to: {matrix_file}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
