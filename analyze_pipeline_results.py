"""
Analyze results from downstream task experiments.

This script:
1. Loads results from all runs
2. Computes comparative statistics
3. Generates tables and plots for the paper
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load_run_results(run_dir):
    """Load results from a single run."""
    # Load final metrics
    metrics_path = os.path.join(run_dir, 'final_metrics.json')
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Load config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    
    # Load history
    history_path = os.path.join(run_dir, 'history.json')
    with open(history_path) as f:
        history = json.load(f)
    
    return {
        'run_name': config['run_name'],
        'train_data': config['train_data'],
        'test_data': config['test_data'],
        'model_type': config['model_type'],
        **metrics,
        'history': history
    }


def analyze_rq1_data_utility(results_df):
    """RQ1: Can synthetic traces replace real data?"""
    print("\n" + "="*60)
    print("RQ1: Data Utility Analysis")
    print("="*60)
    
    # Define configurations
    configs = {
        'Real (Baseline)': 'real_baseline',
        'Synthetic (Raw)': 'synthetic_raw_1024',
        'Synthetic (Repaired)': 'synthetic_repaired_1024',
        'Real + Synthetic': 'combined_1024',
    }
    
    # Extract metrics
    table_data = []
    for label, run_name in configs.items():
        row = results_df[results_df['run_name'] == run_name]
        if len(row) > 0:
            row = row.iloc[0]
            table_data.append({
                'Configuration': label,
                'F1 (Macro)': f"{row['f1_macro']:.4f}",
                'F1 (Weighted)': f"{row['f1_weighted']:.4f}",
                'Accuracy': f"{row['accuracy']:.4f}",
                'Top-5 Acc': f"{row['top5_accuracy']:.4f}",
            })
    
    rq1_df = pd.DataFrame(table_data)
    print("\n", rq1_df.to_string(index=False))
    
    # Compute improvement from repair
    if 'synthetic_raw_1024' in results_df['run_name'].values and 'synthetic_repaired_1024' in results_df['run_name'].values:
        raw_f1 = results_df[results_df['run_name'] == 'synthetic_raw_1024']['f1_macro'].values[0]
        repaired_f1 = results_df[results_df['run_name'] == 'synthetic_repaired_1024']['f1_macro'].values[0]
        improvement = repaired_f1 - raw_f1
        print(f"\n✓ Repair Improvement: ΔF1 = {improvement:.4f} ({improvement/raw_f1*100:.1f}% relative)")
    
    # Compute gap from baseline
    if 'real_baseline' in results_df['run_name'].values and 'synthetic_repaired_1024' in results_df['run_name'].values:
        baseline_f1 = results_df[results_df['run_name'] == 'real_baseline']['f1_macro'].values[0]
        synth_f1 = results_df[results_df['run_name'] == 'synthetic_repaired_1024']['f1_macro'].values[0]
        gap = baseline_f1 - synth_f1
        retention = synth_f1 / baseline_f1
        print(f"✓ Synthetic Data Retention: {retention*100:.1f}% of baseline F1")
        print(f"  (Gap: {gap:.4f})")
    
    return rq1_df


def analyze_rq2_repair_effectiveness(results_df):
    """RQ2: Does repair improve performance?"""
    print("\n" + "="*60)
    print("RQ2: Repair Effectiveness Analysis")
    print("="*60)
    
    # Compare raw vs repaired for each context length
    contexts = ['256', '1024', '4096']
    
    table_data = []
    for ctx in contexts:
        raw_name = f'synthetic_raw_{ctx}'
        repaired_name = f'synthetic_repaired_{ctx}'
        
        raw_row = results_df[results_df['run_name'] == raw_name]
        repaired_row = results_df[results_df['run_name'] == repaired_name]
        
        if len(raw_row) > 0 and len(repaired_row) > 0:
            raw_f1 = raw_row.iloc[0]['f1_macro']
            repaired_f1 = repaired_row.iloc[0]['f1_macro']
            delta = repaired_f1 - raw_f1
            
            table_data.append({
                'Context Length': ctx,
                'Raw F1': f"{raw_f1:.4f}",
                'Repaired F1': f"{repaired_f1:.4f}",
                'ΔF1': f"{delta:.4f}",
                'Improvement': f"{delta/raw_f1*100:.1f}%"
            })
    
    rq2_df = pd.DataFrame(table_data)
    print("\n", rq2_df.to_string(index=False))
    
    return rq2_df


def analyze_rq3_context_length(results_df):
    """RQ3: Does context length matter?"""
    print("\n" + "="*60)
    print("RQ3: Context Length Impact Analysis")
    print("="*60)
    
    contexts = ['256', '1024', '4096']
    
    table_data = []
    for ctx in contexts:
        run_name = f'synthetic_repaired_{ctx}'
        row = results_df[results_df['run_name'] == run_name]
        
        if len(row) > 0:
            row = row.iloc[0]
            table_data.append({
                'Context Length': ctx,
                'F1 (Macro)': f"{row['f1_macro']:.4f}",
                'Accuracy': f"{row['accuracy']:.4f}",
                'Top-5 Acc': f"{row['top5_accuracy']:.4f}",
            })
    
    rq3_df = pd.DataFrame(table_data)
    print("\n", rq3_df.to_string(index=False))
    
    # Compute improvement from 256 to 4096
    if len(table_data) >= 2:
        f1_256 = results_df[results_df['run_name'] == 'synthetic_repaired_256']['f1_macro'].values
        f1_4096 = results_df[results_df['run_name'] == 'synthetic_repaired_4096']['f1_macro'].values
        
        if len(f1_256) > 0 and len(f1_4096) > 0:
            improvement = f1_4096[0] - f1_256[0]
            print(f"\n✓ Context Length Improvement (256→4096): ΔF1 = {improvement:.4f}")
    
    return rq3_df


def analyze_rq4_feature_ablation(results_df):
    """RQ4: Which features are critical?"""
    print("\n" + "="*60)
    print("RQ4: Feature Ablation Analysis")
    print("="*60)
    
    # This requires running experiments with different channel configurations
    # For now, show event-only vs full
    
    configs = {
        'Event Only': 'event_only_baseline',
        'Full Features': 'real_baseline',
    }
    
    table_data = []
    for label, run_name in configs.items():
        row = results_df[results_df['run_name'] == run_name]
        if len(row) > 0:
            row = row.iloc[0]
            table_data.append({
                'Configuration': label,
                'F1 (Macro)': f"{row['f1_macro']:.4f}",
                'Accuracy': f"{row['accuracy']:.4f}",
            })
    
    if len(table_data) > 0:
        rq4_df = pd.DataFrame(table_data)
        print("\n", rq4_df.to_string(index=False))
        return rq4_df
    else:
        print("\n[Note] Feature ablation experiments not found. Run with --model-type event_only")
        return None


def generate_summary_table(results_df, output_path):
    """Generate comprehensive summary table."""
    print("\n" + "="*60)
    print("Summary Table (All Runs)")
    print("="*60)
    
    summary_data = []
    for _, row in results_df.iterrows():
        summary_data.append({
            'Run Name': row['run_name'],
            'F1 (Macro)': f"{row['f1_macro']:.4f}",
            'F1 (Weighted)': f"{row['f1_weighted']:.4f}",
            'Accuracy': f"{row['accuracy']:.4f}",
            'Top-5 Acc': f"{row['top5_accuracy']:.4f}",
            'Top-10 Acc': f"{row['top10_accuracy']:.4f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    print(f"\n[Saved] {output_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='experiments_downstream/results')
    parser.add_argument('--output-dir', default='experiments_downstream/analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all results
    print("[Loading] Results from all runs...")
    results = []
    
    for run_dir in Path(args.results_dir).iterdir():
        if run_dir.is_dir():
            result = load_run_results(run_dir)
            if result:
                results.append(result)
                print(f"  ✓ {result['run_name']}")
    
    if not results:
        print("[Error] No results found!")
        return
    
    # Create DataFrame
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'history'} for r in results])
    
    # Generate summary
    summary_df = generate_summary_table(
        results_df,
        os.path.join(args.output_dir, 'summary_all_runs.csv')
    )
    
    # Analyze each RQ
    rq1_df = analyze_rq1_data_utility(results_df)
    if rq1_df is not None:
        rq1_df.to_csv(os.path.join(args.output_dir, 'rq1_data_utility.csv'), index=False)
    
    rq2_df = analyze_rq2_repair_effectiveness(results_df)
    if rq2_df is not None:
        rq2_df.to_csv(os.path.join(args.output_dir, 'rq2_repair_effectiveness.csv'), index=False)
    
    rq3_df = analyze_rq3_context_length(results_df)
    if rq3_df is not None:
        rq3_df.to_csv(os.path.join(args.output_dir, 'rq3_context_length.csv'), index=False)
    
    rq4_df = analyze_rq4_feature_ablation(results_df)
    if rq4_df is not None:
        rq4_df.to_csv(os.path.join(args.output_dir, 'rq4_feature_ablation.csv'), index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nKey Findings:")
    print("1. Check rq1_data_utility.csv for synthetic data utility")
    print("2. Check rq2_repair_effectiveness.csv for repair impact")
    print("3. Check rq3_context_length.csv for context length analysis")
    print("4. Check rq4_feature_ablation.csv for feature importance")


if __name__ == '__main__':
    main()
