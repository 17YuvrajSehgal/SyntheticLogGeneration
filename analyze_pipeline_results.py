#!/usr/bin/env python3
"""
Comprehensive Pipeline Results Analyzer

Collects and analyzes results from run_pipeline.py across multiple benchmarks.
Generates comparative tables and summary statistics.

Usage:
    python analyze_pipeline_results.py
    python analyze_pipeline_results.py --benchmarks ffmpeg pybench scimark2
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def load_experiment_metrics(experiment_dir):
    """Load metrics from a single experiment directory."""
    metrics_file = experiment_dir / 'final_metrics.json'
    config_file = experiment_dir / 'config.json'
    
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON error in {metrics_file}: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Error reading {metrics_file}: {e}")
        return None
    
    config = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except:
            pass  # Config is optional
    
    return {
        'run_name': experiment_dir.name,
        'config': config,
        **metrics
    }


def parse_run_name(run_name):
    """Parse run name to extract configuration details."""
    # Format: {config}_{benchmark}_{window}
    # Examples: real_baseline_ffmpeg_1024, combined_50_50_ffmpeg_1024
    
    parts = run_name.split('_')
    
    # Extract benchmark (second to last part)
    benchmark = parts[-2] if len(parts) >= 2 else 'unknown'
    
    # Extract window (last part)
    try:
        window = int(parts[-1])
    except:
        window = None
    
    # Extract configuration type
    if 'real_baseline' in run_name:
        config_type = 'Real Only'
    elif 'synthetic_data_only' in run_name:
        config_type = 'Synthetic Only'
    elif 'combined_50_50_norepair' in run_name:
        config_type = 'Combined (No Repair)'
    elif 'combined_50_50' in run_name:
        config_type = 'Combined (Repaired)'
    else:
        config_type = 'Unknown'
    
    return {
        'benchmark': benchmark,
        'window': window,
        'config_type': config_type
    }


def collect_all_results(base_dir, benchmarks):
    """Collect all results from specified benchmarks."""
    all_results = []
    
    for benchmark in benchmarks:
        benchmark_dir = Path(base_dir) / benchmark
        
        if not benchmark_dir.exists():
            print(f"⚠️  Benchmark not found: {benchmark}")
            continue
        
        print(f"\n📊 Collecting results for: {benchmark}")
        
        # Check all window sizes
        for window_dir in benchmark_dir.iterdir():
            if not window_dir.is_dir():
                continue
            
            results_dir = window_dir / 'results'
            if not results_dir.exists():
                continue
            
            # Load all experiments in this results directory
            for exp_dir in results_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                
                metrics = load_experiment_metrics(exp_dir)
                if metrics:
                    # Parse run name
                    parsed = parse_run_name(metrics['run_name'])
                    metrics.update(parsed)
                    all_results.append(metrics)
                    print(f"  ✅ {metrics['run_name']}")
    
    return all_results


def create_summary_table(results_df, output_dir):
    """Create comprehensive summary table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*80)
    
    summary_data = []
    for _, row in results_df.iterrows():
        summary_data.append({
            'Benchmark': row['benchmark'],
            'Window': row['window'],
            'Configuration': row['config_type'],
            'F1 (Macro)': row['f1_macro'],
            'F1 (Weighted)': row['f1_weighted'],
            'Accuracy': row['accuracy'],
            'Top-5 Acc': row['top5_accuracy'],
            'Top-10 Acc': row['top10_accuracy'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by benchmark, window, config
    summary_df = summary_df.sort_values(['Benchmark', 'Window', 'Configuration'])
    
    # Format for display
    display_df = summary_df.copy()
    for col in ['F1 (Macro)', 'F1 (Weighted)', 'Accuracy', 'Top-5 Acc', 'Top-10 Acc']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
    
    print("\n", display_df.to_string(index=False))
    
    # Save to CSV
    output_file = output_dir / 'summary_all_results.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    return summary_df


def create_benchmark_comparison(summary_df, output_dir):
    """Create benchmark comparison table (1024 window only)."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON (Window=1024)")
    print("="*80)
    
    # Filter for 1024 window
    df_1024 = summary_df[summary_df['Window'] == 1024].copy()
    
    if df_1024.empty:
        print("⚠️  No results for window=1024")
        return None
    
    # Pivot table: benchmarks vs configurations
    pivot = df_1024.pivot_table(
        index='Benchmark',
        columns='Configuration',
        values='F1 (Macro)',  # Use title case from summary_df
        aggfunc='first'
    )
    
    # Reorder columns
    col_order = ['Real Only', 'Synthetic Only', 'Combined (No Repair)', 'Combined (Repaired)']
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    
    # Format as percentages
    pivot_pct = pivot * 100
    
    print("\n", pivot_pct.to_string(float_format=lambda x: f'{x:.2f}%' if pd.notna(x) else '-'))
    
    # Save to CSV
    output_file = output_dir / 'benchmark_comparison_1024.csv'
    pivot.to_csv(output_file)
    print(f"\n✅ Saved: {output_file}")
    
    return pivot


def analyze_data_augmentation_benefit(summary_df, output_dir):
    """Analyze data augmentation benefit across benchmarks."""
    print("\n" + "="*80)
    print("DATA AUGMENTATION BENEFIT ANALYSIS")
    print("="*80)
    
    analysis_data = []
    
    for benchmark in summary_df['Benchmark'].unique():
        for window in summary_df['Window'].unique():
            # Get relevant rows
            subset = summary_df[
                (summary_df['Benchmark'] == benchmark) &
                (summary_df['Window'] == window)
            ]
            
            real_only = subset[subset['Configuration'] == 'Real Only']
            combined = subset[subset['Configuration'] == 'Combined (Repaired)']
            
            if not real_only.empty and not combined.empty:
                real_f1 = real_only.iloc[0]['F1 (Macro)']
                combined_f1 = combined.iloc[0]['F1 (Macro)']
                improvement = combined_f1 - real_f1
                improvement_pct = (improvement / real_f1) * 100
                
                analysis_data.append({
                    'Benchmark': benchmark,
                    'Window': window,
                    'Real Only F1': real_f1,
                    'Combined F1': combined_f1,
                    'Improvement (Δ)': improvement,
                    'Improvement (%)': improvement_pct
                })
    
    if not analysis_data:
        print("⚠️  Insufficient data for augmentation analysis")
        return None
    
    aug_df = pd.DataFrame(analysis_data)
    aug_df = aug_df.sort_values(['Benchmark', 'Window'])
    
    # Format for display
    display_df = aug_df.copy()
    display_df['Real Only F1'] = display_df['Real Only F1'].apply(lambda x: f'{x:.4f}')
    display_df['Combined F1'] = display_df['Combined F1'].apply(lambda x: f'{x:.4f}')
    display_df['Improvement (Δ)'] = display_df['Improvement (Δ)'].apply(lambda x: f'{x:+.4f}')
    display_df['Improvement (%)'] = display_df['Improvement (%)'].apply(lambda x: f'{x:+.2f}%')
    
    print("\n", display_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    print(f"Average Improvement: {aug_df['Improvement (Δ)'].mean():+.4f} ({aug_df['Improvement (%)'].mean():+.2f}%)")
    print(f"Best Improvement: {aug_df['Improvement (Δ)'].max():+.4f} ({aug_df.loc[aug_df['Improvement (Δ)'].idxmax(), 'Benchmark']} @ {aug_df.loc[aug_df['Improvement (Δ)'].idxmax(), 'Window']})")
    print(f"Worst Improvement: {aug_df['Improvement (Δ)'].min():+.4f} ({aug_df.loc[aug_df['Improvement (Δ)'].idxmin(), 'Benchmark']} @ {aug_df.loc[aug_df['Improvement (Δ)'].idxmin(), 'Window']})")
    
    # Save to CSV
    output_file = output_dir / 'augmentation_benefit.csv'
    aug_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    return aug_df


def analyze_repair_effectiveness(summary_df, output_dir):
    """Analyze constraint repair effectiveness."""
    print("\n" + "="*80)
    print("CONSTRAINT REPAIR EFFECTIVENESS")
    print("="*80)
    
    analysis_data = []
    
    for benchmark in summary_df['Benchmark'].unique():
        for window in summary_df['Window'].unique():
            subset = summary_df[
                (summary_df['Benchmark'] == benchmark) &
                (summary_df['Window'] == window)
            ]
            
            no_repair = subset[subset['Configuration'] == 'Combined (No Repair)']
            repaired = subset[subset['Configuration'] == 'Combined (Repaired)']
            
            if not no_repair.empty and not repaired.empty:
                no_repair_f1 = no_repair.iloc[0]['F1 (Macro)']
                repaired_f1 = repaired.iloc[0]['F1 (Macro)']
                improvement = repaired_f1 - no_repair_f1
                improvement_pct = (improvement / no_repair_f1) * 100
                
                analysis_data.append({
                    'Benchmark': benchmark,
                    'Window': window,
                    'No Repair F1': no_repair_f1,
                    'Repaired F1': repaired_f1,
                    'Improvement (Δ)': improvement,
                    'Improvement (%)': improvement_pct
                })
    
    if not analysis_data:
        print("⚠️  Insufficient data for repair analysis")
        return None
    
    repair_df = pd.DataFrame(analysis_data)
    repair_df = repair_df.sort_values(['Benchmark', 'Window'])
    
    # Format for display
    display_df = repair_df.copy()
    display_df['No Repair F1'] = display_df['No Repair F1'].apply(lambda x: f'{x:.4f}')
    display_df['Repaired F1'] = display_df['Repaired F1'].apply(lambda x: f'{x:.4f}')
    display_df['Improvement (Δ)'] = display_df['Improvement (Δ)'].apply(lambda x: f'{x:+.4f}')
    display_df['Improvement (%)'] = display_df['Improvement (%)'].apply(lambda x: f'{x:+.2f}%')
    
    print("\n", display_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    print(f"Average Improvement: {repair_df['Improvement (Δ)'].mean():+.4f} ({repair_df['Improvement (%)'].mean():+.2f}%)")
    print(f"Best Improvement: {repair_df['Improvement (Δ)'].max():+.4f} ({repair_df.loc[repair_df['Improvement (Δ)'].idxmax(), 'Benchmark']} @ {repair_df.loc[repair_df['Improvement (Δ)'].idxmax(), 'Window']})")
    
    # Save to CSV
    output_file = output_dir / 'repair_effectiveness.csv'
    repair_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    return repair_df


def analyze_context_length_impact(summary_df, output_dir):
    """Analyze impact of context length."""
    print("\n" + "="*80)
    print("CONTEXT LENGTH IMPACT ANALYSIS")
    print("="*80)
    
    # Focus on Combined (Repaired) configuration
    df_combined = summary_df[summary_df['Configuration'] == 'Combined (Repaired)'].copy()
    
    if df_combined.empty:
        print("⚠️  No Combined (Repaired) results found")
        return None
    
    # Pivot: benchmarks vs window sizes
    pivot = df_combined.pivot_table(
        index='Benchmark',
        columns='Window',
        values='F1 (Macro)',  # Use title case from summary_df
        aggfunc='first'
    )
    
    # Format as percentages
    pivot_pct = pivot * 100
    
    print("\n", pivot_pct.to_string(float_format=lambda x: f'{x:.2f}%' if pd.notna(x) else '-'))
    
    # Calculate improvement from 256 to 4096
    if 256 in pivot.columns and 4096 in pivot.columns:
        pivot['Improvement (256→4096)'] = (pivot[4096] - pivot[256]) * 100
        print("\n" + "-"*80)
        print("Improvement from 256 to 4096:")
        print("-"*80)
        for benchmark in pivot.index:
            if pd.notna(pivot.loc[benchmark, 'Improvement (256→4096)']):
                print(f"{benchmark}: {pivot.loc[benchmark, 'Improvement (256→4096)']:+.2f}%")
    
    # Save to CSV
    output_file = output_dir / 'context_length_impact.csv'
    pivot.to_csv(output_file)
    print(f"\n✅ Saved: {output_file}")
    
    return pivot


def generate_latex_table(summary_df, output_dir):
    """Generate LaTeX table for paper."""
    print("\n" + "="*80)
    print("GENERATING LATEX TABLE")
    print("="*80)
    
    # Filter for 1024 window
    df_1024 = summary_df[summary_df['Window'] == 1024].copy()
    
    if df_1024.empty:
        print("⚠️  No results for window=1024")
        return
    
    # Pivot table
    pivot = df_1024.pivot_table(
        index='Benchmark',
        columns='Configuration',
        values='F1 (Macro)',  # Use title case from summary_df
        aggfunc='first'
    )
    
    # Reorder columns
    col_order = ['Real Only', 'Synthetic Only', 'Combined (No Repair)', 'Combined (Repaired)']
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    
    # Generate LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Downstream Task Performance (F1 Macro) for Next-Event Prediction}")
    latex_lines.append("\\label{tab:downstream-results}")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
    latex_lines.append("\\toprule")
    
    # Header
    header = "Benchmark & " + " & ".join(pivot.columns) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Data rows
    for benchmark in pivot.index:
        row_data = [f"{pivot.loc[benchmark, col]*100:.2f}\\%" if pd.notna(pivot.loc[benchmark, col]) else "-" 
                   for col in pivot.columns]
        row = f"{benchmark} & " + " & ".join(row_data) + " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    print("\n", latex_content)
    
    # Save to file
    output_file = output_dir / 'results_table.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    print(f"\n✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze pipeline results across benchmarks')
    parser.add_argument('--base-dir', default='experiments_downstream_results',
                       help='Base directory containing results')
    parser.add_argument('--benchmarks', nargs='+', 
                       default=['ffmpeg', 'pybench', 'scimark2', 'stream', 'unpack-linux'],
                       help='Benchmarks to analyze')
    parser.add_argument('--output-dir', default='experiments_downstream_results/results-pipeline',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PIPELINE RESULTS ANALYZER")
    print("="*80)
    print(f"Base Directory: {args.base_dir}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80)
    
    # Collect all results
    results = collect_all_results(args.base_dir, args.benchmarks)
    
    if not results:
        print("\n❌ No results found!")
        return
    
    print(f"\n✅ Collected {len(results)} experiment results")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate summary (this creates proper column names)
    summary_df = create_summary_table(results_df, output_dir)
    
    # Generate analyses using summary_df
    benchmark_comp = create_benchmark_comparison(summary_df, output_dir)
    aug_benefit = analyze_data_augmentation_benefit(summary_df, output_dir)
    repair_effect = analyze_repair_effectiveness(summary_df, output_dir)
    context_impact = analyze_context_length_impact(summary_df, output_dir)
    generate_latex_table(summary_df, output_dir)
    
    # Generate summary report
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. summary_all_results.csv - Complete results table")
    print("  2. benchmark_comparison_1024.csv - Benchmark comparison (1024 window)")
    print("  3. augmentation_benefit.csv - Data augmentation analysis")
    print("  4. repair_effectiveness.csv - Constraint repair analysis")
    print("  5. context_length_impact.csv - Context length analysis")
    print("  6. results_table.tex - LaTeX table for paper")
    
    # Key findings
    if aug_benefit is not None and not aug_benefit.empty:
        avg_improvement = aug_benefit['Improvement (%)'].mean()
        print(f"\n🎯 Key Finding: Data augmentation improves F1 by {avg_improvement:+.2f}% on average")
    
    if repair_effect is not None and not repair_effect.empty:
        avg_repair = repair_effect['Improvement (%)'].mean()
        print(f"🔧 Key Finding: Constraint repair improves F1 by {avg_repair:+.2f}% on average")


if __name__ == '__main__':
    main()
