#!/usr/bin/env python3
"""
Comprehensive Ablation Study Results Collector

Collects all results from cross-evaluation experiments and creates:
1. Summary table (markdown)
2. Detailed metrics (CSV)
3. Combined JSON report

Usage:
    python collect_ablation_results.py --benchmark scimark2
    python collect_ablation_results.py --benchmark ffmpeg --output-dir ./summary
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime


def load_metrics(result_dir):
    """Load final_metrics.json from a result directory."""
    metrics_file = os.path.join(result_dir, 'final_metrics.json')
    config_file = os.path.join(result_dir, 'config.json')
    
    if not os.path.exists(metrics_file):
        return None, None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    config = None
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    return metrics, config


def parse_experiment_name(run_name):
    """Parse experiment name to extract diffusion model and predictor channels."""
    # Format: cross_{diffusion}_{predictor}
    # Examples: cross_base_event, cross_full_full, cross_system_system
    
    parts = run_name.replace('cross_', '').split('_')
    
    if len(parts) < 2:
        return None, None
    
    diffusion = parts[0]  # base, system, full
    predictor = '_'.join(parts[1:])  # event, base, system, full
    
    # Map predictor names to channel descriptions
    predictor_map = {
        'event': 'event',
        'base': 'event+dt',
        'system': 'event+dt+cpu+tid',
        'full': 'all 6'
    }
    
    predictor_desc = predictor_map.get(predictor, predictor)
    
    return diffusion.capitalize(), predictor_desc


def collect_all_results(base_dir):
    """Collect all results from cross-results directory."""
    results = []
    
    # Expected experiment names
    experiments = [
        'cross_base_event', 'cross_base_base',
        'cross_system_event', 'cross_system_base', 'cross_system_system',
        'cross_full_event', 'cross_full_base', 'cross_full_system', 'cross_full_full'
    ]
    
    for exp_name in experiments:
        result_dir = os.path.join(base_dir, exp_name)
        
        if not os.path.exists(result_dir):
            print(f"⚠️  Missing: {exp_name}")
            continue
        
        metrics, config = load_metrics(result_dir)
        
        if metrics is None:
            print(f"⚠️  No metrics: {exp_name}")
            continue
        
        diffusion, predictor = parse_experiment_name(exp_name)
        
        result = {
            'Experiment': exp_name,
            'Diffusion Model': diffusion,
            'Predictor Channels': predictor,
            'F1 (Macro)': metrics.get('f1_macro', 0.0),
            'F1 (Weighted)': metrics.get('f1_weighted', 0.0),
            'Accuracy': metrics.get('accuracy', 0.0),
            'Top-5 Acc': metrics.get('top5_accuracy', 0.0),
            'Top-10 Acc': metrics.get('top10_accuracy', 0.0),
            'Loss': metrics.get('loss', 0.0),
        }
        
        # Add config info if available
        if config:
            result['Epochs'] = config.get('epochs', 'N/A')
            result['Batch Size'] = config.get('batch_size', 'N/A')
            result['Seq Len'] = config.get('seq_len', 'N/A')
        
        results.append(result)
        print(f"✅ Loaded: {exp_name}")
    
    return pd.DataFrame(results)


def create_pivot_table(df):
    """Create pivot table for cross-evaluation matrix."""
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


def generate_summary_report(df, benchmark, output_dir):
    """Generate comprehensive summary report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"ABLATION STUDY RESULTS SUMMARY: {benchmark.upper()}")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Experiments: {len(df)}")
    report_lines.append("=" * 80)
    
    # Overall statistics
    report_lines.append("\n## OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Best F1 (Macro): {df['F1 (Macro)'].max():.4f} ({df.loc[df['F1 (Macro)'].idxmax(), 'Experiment']})")
    report_lines.append(f"Worst F1 (Macro): {df['F1 (Macro)'].min():.4f} ({df.loc[df['F1 (Macro)'].idxmin(), 'Experiment']})")
    report_lines.append(f"Mean F1 (Macro): {df['F1 (Macro)'].mean():.4f}")
    report_lines.append(f"Std F1 (Macro): {df['F1 (Macro)'].std():.4f}")
    
    # Cross-evaluation matrix
    report_lines.append("\n## CROSS-EVALUATION MATRIX (F1 Macro %)")
    report_lines.append("-" * 80)
    pivot = create_pivot_table(df)
    pivot_pct = pivot * 100
    report_lines.append(pivot_pct.to_string(float_format=lambda x: f'{x:.2f}%' if pd.notna(x) else '-'))
    
    # Detailed results table
    report_lines.append("\n## DETAILED RESULTS")
    report_lines.append("-" * 80)
    display_cols = ['Diffusion Model', 'Predictor Channels', 'F1 (Macro)', 'Accuracy', 'F1 (Weighted)', 'Top-5 Acc']
    report_lines.append(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Key findings
    report_lines.append("\n## KEY FINDINGS")
    report_lines.append("-" * 80)
    
    # Diagonal performance
    diagonal_exps = ['cross_base_base', 'cross_system_system', 'cross_full_full']
    diagonal_df = df[df['Experiment'].isin(diagonal_exps)]
    if not diagonal_df.empty:
        report_lines.append("\n1. Diagonal (Matching Configurations):")
        for _, row in diagonal_df.iterrows():
            report_lines.append(f"   {row['Diffusion Model']} → {row['Predictor Channels']}: {row['F1 (Macro)']*100:.2f}%")
    
    # Best per diffusion model
    report_lines.append("\n2. Best Predictor per Diffusion Model:")
    for diffusion in ['Base', 'System', 'Full']:
        subset = df[df['Diffusion Model'] == diffusion]
        if not subset.empty:
            best = subset.loc[subset['F1 (Macro)'].idxmax()]
            report_lines.append(f"   {diffusion}: {best['Predictor Channels']} ({best['F1 (Macro)']*100:.2f}%)")
    
    # Best per predictor
    report_lines.append("\n3. Best Diffusion Model per Predictor:")
    for predictor in ['event', 'event+dt', 'event+dt+cpu+tid', 'all 6']:
        subset = df[df['Predictor Channels'] == predictor]
        if not subset.empty:
            best = subset.loc[subset['F1 (Macro)'].idxmax()]
            report_lines.append(f"   {predictor}: {best['Diffusion Model']} ({best['F1 (Macro)']*100:.2f}%)")
    
    report_lines.append("\n" + "=" * 80)
    
    # Save report
    report_file = os.path.join(output_dir, f'ablation_summary_{benchmark}.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✅ Summary report saved: {report_file}")
    
    # Print to console
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description='Collect ablation study results')
    parser.add_argument('--benchmark', required=True, help='Benchmark name')
    parser.add_argument('--base-dir', default='$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation-diffusion',
                       help='Base directory for results')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same as cross-results)')
    
    args = parser.parse_args()
    
    # Expand environment variables
    base_dir = os.path.expandvars(args.base_dir)
    cross_results_dir = os.path.join(base_dir, args.benchmark, 'cross-results')
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = cross_results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Collecting Ablation Results: {args.benchmark}")
    print(f"{'='*80}")
    print(f"Source: {cross_results_dir}")
    print(f"Output: {output_dir}\n")
    
    # Check if directory exists
    if not os.path.exists(cross_results_dir):
        print(f"❌ Results directory not found: {cross_results_dir}")
        return
    
    # Collect results
    df = collect_all_results(cross_results_dir)
    
    if df.empty:
        print("❌ No results found!")
        return
    
    # Save detailed CSV
    csv_file = os.path.join(output_dir, f'ablation_detailed_{args.benchmark}.csv')
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Detailed CSV saved: {csv_file}")
    
    # Save pivot table
    pivot = create_pivot_table(df)
    pivot_file = os.path.join(output_dir, f'ablation_matrix_{args.benchmark}.csv')
    pivot.to_csv(pivot_file)
    print(f"✅ Matrix CSV saved: {pivot_file}")
    
    # Save combined JSON
    json_file = os.path.join(output_dir, f'ablation_complete_{args.benchmark}.json')
    with open(json_file, 'w') as f:
        json.dump({
            'benchmark': args.benchmark,
            'timestamp': datetime.now().isoformat(),
            'results': df.to_dict('records'),
            'matrix': pivot.to_dict()
        }, f, indent=2)
    print(f"✅ JSON report saved: {json_file}")
    
    # Generate summary report
    generate_summary_report(df, args.benchmark, output_dir)


if __name__ == '__main__':
    main()
