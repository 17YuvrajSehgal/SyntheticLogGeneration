#!/usr/bin/env python3
"""
Privacy Analysis: Nearest Neighbor Distance Analysis for Synthetic Traces
Determines if the model is generating novel traces or memorizing training data
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


def load_npz_windows(file_path, max_windows=None):
    """Load windows from NPZ file."""
    data = np.load(file_path)
    event = data['event']  # [N, L]
    dt = data['dt']
    cpu = data['cpu']

    if max_windows is not None:
        event = event[:max_windows]
        dt = dt[:max_windows]
        cpu = cpu[:max_windows]

    # Stack into [N, L, 3]
    windows = np.stack([event, dt, cpu], axis=-1)
    return windows


def compute_window_embedding(window):
    """
    Compute a feature embedding for a window.
    Multiple strategies can be used:
    1. Flatten the window (simple but high-dimensional)
    2. Statistical features (mean, std, histogram)
    3. N-gram features
    """
    # Strategy: Use statistical features for more robust comparison
    event_seq = window[:, 0]
    dt_seq = window[:, 1]
    cpu_seq = window[:, 2]

    features = []

    # Basic statistics for each channel
    features.extend([
        np.mean(event_seq), np.std(event_seq), np.median(event_seq),
        np.mean(dt_seq), np.std(dt_seq), np.median(dt_seq),
        np.mean(cpu_seq), np.std(cpu_seq), np.median(cpu_seq),
    ])

    # Event type distribution (histogram)
    event_hist, _ = np.histogram(event_seq, bins=50, range=(0, 380))
    features.extend(event_hist / (len(event_seq) + 1e-8))

    # Delta time distribution
    dt_hist, _ = np.histogram(dt_seq, bins=20, range=(0, 256))
    features.extend(dt_hist / (len(dt_seq) + 1e-8))

    # CPU distribution
    cpu_hist, _ = np.histogram(cpu_seq, bins=4, range=(0, 4))
    features.extend(cpu_hist / (len(cpu_seq) + 1e-8))

    # Transition features (bigrams)
    event_transitions = np.column_stack([event_seq[:-1], event_seq[1:]])
    unique_transitions = len(np.unique(event_transitions, axis=0))
    features.append(unique_transitions / len(event_seq))

    return np.array(features, dtype=np.float32)


def compute_embeddings_batch(windows):
    """Compute embeddings for a batch of windows."""
    embeddings = np.array([compute_window_embedding(w) for w in windows])
    return embeddings


def euclidean_distance_matrix(embeddings1, embeddings2):
    """Compute pairwise Euclidean distances."""
    return cdist(embeddings1, embeddings2, metric='euclidean')


def cosine_distance_matrix(embeddings1, embeddings2):
    """Compute pairwise cosine distances."""
    return cdist(embeddings1, embeddings2, metric='cosine')


def compute_nearest_neighbor_distances(synthetic_embeds, real_embeds, 
                                       distance_metric='euclidean', 
                                       batch_size=1000):
    """
    Compute nearest neighbor distance for each synthetic window to real windows.
    Returns array of minimum distances for each synthetic sample.
    """
    n_synthetic = len(synthetic_embeds)
    nn_distances = np.zeros(n_synthetic)

    print(f"Computing {distance_metric} nearest neighbor distances...")

    for i in tqdm(range(0, n_synthetic, batch_size)):
        end_idx = min(i + batch_size, n_synthetic)
        batch_embeds = synthetic_embeds[i:end_idx]

        if distance_metric == 'euclidean':
            dist_matrix = euclidean_distance_matrix(batch_embeds, real_embeds)
        elif distance_metric == 'cosine':
            dist_matrix = cosine_distance_matrix(batch_embeds, real_embeds)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        # Find minimum distance for each synthetic sample
        nn_distances[i:end_idx] = np.min(dist_matrix, axis=1)

    return nn_distances


def compute_exact_matches(synthetic_windows, real_windows):
    """
    Count exact matches between synthetic and real windows.
    This is a strong indicator of memorization.
    """
    print("Checking for exact matches...")

    # Flatten windows for comparison
    synth_flat = synthetic_windows.reshape(len(synthetic_windows), -1)
    real_flat = real_windows.reshape(len(real_windows), -1)

    # Convert to a set of tuples for faster lookup
    real_set = set(tuple(row) for row in real_flat)

    exact_matches = 0
    for synth_row in tqdm(synth_flat):
        if tuple(synth_row) in real_set:
            exact_matches += 1

    return exact_matches


def plot_distance_distribution(nn_distances, output_path, title_suffix=""):
    """Create visualization of nearest neighbor distance distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(nn_distances, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Nearest Neighbor Distance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of NN Distances {title_suffix}')
    axes[0, 0].axvline(np.median(nn_distances), color='red', 
                       linestyle='--', label=f'Median: {np.median(nn_distances):.3f}')
    axes[0, 0].legend()

    # CDF
    sorted_distances = np.sort(nn_distances)
    cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    axes[0, 1].plot(sorted_distances, cdf, linewidth=2)
    axes[0, 1].set_xlabel('Nearest Neighbor Distance')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title(f'CDF of NN Distances {title_suffix}')
    axes[0, 1].grid(True, alpha=0.3)

    # Box plot
    axes[1, 0].boxplot(nn_distances, vert=False)
    axes[1, 0].set_xlabel('Nearest Neighbor Distance')
    axes[1, 0].set_title(f'Box Plot of NN Distances {title_suffix}')
    axes[1, 0].grid(True, alpha=0.3)

    # Log-scale histogram
    axes[1, 1].hist(nn_distances, bins=100, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Nearest Neighbor Distance')
    axes[1, 1].set_ylabel('Frequency (log scale)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title(f'Log-Scale Distribution {title_suffix}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def compute_statistics(nn_distances):
    """Compute summary statistics for nearest neighbor distances."""
    stats = {
        'mean': np.mean(nn_distances),
        'median': np.median(nn_distances),
        'std': np.std(nn_distances),
        'min': np.min(nn_distances),
        'max': np.max(nn_distances),
        'q25': np.percentile(nn_distances, 25),
        'q75': np.percentile(nn_distances, 75),
        'q95': np.percentile(nn_distances, 95),
        'q99': np.percentile(nn_distances, 99),
        'very_close_pct': np.mean(nn_distances < 0.1) * 100,  # % with distance < 0.1
        'close_pct': np.mean(nn_distances < 0.5) * 100,       # % with distance < 0.5
    }
    return stats


def print_analysis_report(stats_synth_to_real, stats_synth_to_synth, 
                         exact_matches, n_synthetic, n_real):
    """Print comprehensive analysis report."""
    print("\n" + "="*70)
    print("PRIVACY ANALYSIS REPORT: Memorization Detection")
    print("="*70)

    print(f"\nDataset Sizes:")
    print(f"  Synthetic windows: {n_synthetic:,}")
    print(f"  Real windows: {n_real:,}")

    print(f"\n{'SYNTHETIC -> REAL (Privacy Metric)':-^70}")
    print(f"  Mean distance:        {stats_synth_to_real['mean']:.4f}")
    print(f"  Median distance:      {stats_synth_to_real['median']:.4f}")
    print(f"  Std deviation:        {stats_synth_to_real['std']:.4f}")
    print(f"  Min distance:         {stats_synth_to_real['min']:.4f}")
    print(f"  25th percentile:      {stats_synth_to_real['q25']:.4f}")
    print(f"  75th percentile:      {stats_synth_to_real['q75']:.4f}")
    print(f"  95th percentile:      {stats_synth_to_real['q95']:.4f}")
    print(f"  99th percentile:      {stats_synth_to_real['q99']:.4f}")

    print(f"\n{'Distance Thresholds:':-^70}")
    print(f"  % very close (< 0.1): {stats_synth_to_real['very_close_pct']:.2f}%")
    print(f"  % close (< 0.5):      {stats_synth_to_real['close_pct']:.2f}%")

    print(f"\n{'Exact Matches:':-^70}")
    print(f"  Exact duplicates:     {exact_matches} / {n_synthetic}")
    print(f"  Duplication rate:     {(exact_matches / n_synthetic * 100):.4f}%")

    if stats_synth_to_synth is not None:
        print(f"\n{'SYNTHETIC -> SYNTHETIC (Diversity Metric)':-^70}")
        print(f"  Mean distance:        {stats_synth_to_synth['mean']:.4f}")
        print(f"  Median distance:      {stats_synth_to_synth['median']:.4f}")
        print(f"  Min distance:         {stats_synth_to_synth['min']:.4f}")
        print(f"  % very close (< 0.1): {stats_synth_to_synth['very_close_pct']:.2f}%")

    print(f"\n{'INTERPRETATION:':-^70}")

    if exact_matches > 0:
        print("  WARNING: Exact matches found! Model has memorized training data.")
    else:
        print("  OK: No exact matches found.")

    if stats_synth_to_real['median'] < 0.5:
        print("  WARNING: Low median distance suggests potential memorization.")
        print("     Synthetic samples are very similar to real training data.")
    elif stats_synth_to_real['median'] < 1.5:
        print("  MODERATE: Median distance indicates some novelty but samples")
        print("     are still relatively close to training data.")
    else:
        print("  GOOD: High median distance indicates novel synthetic samples.")
        print("     Model is generating diverse, non-memorized data.")

    if stats_synth_to_real['very_close_pct'] > 10:
        print(f"  WARNING: {stats_synth_to_real['very_close_pct']:.1f}% of synthetic samples")
        print("     are very close to real data (distance < 0.1).")

    if stats_synth_to_synth is not None:
        if stats_synth_to_synth['very_close_pct'] > 5:
            print(f"  WARNING: {stats_synth_to_synth['very_close_pct']:.1f}% of synthetic samples")
            print("     are very similar to each other (low diversity).")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze privacy and memorization in synthetic trace data'
    )
    parser.add_argument('--real_glob', type=str, required=True,
                       help='Glob pattern for real data NPZ files (e.g., "window_shards/compress-gzip/train/*.npz")')
    parser.add_argument('--synth', type=str, required=True,
                       help='Path to synthetic data NPZ file')
    parser.add_argument('--max_real_windows', type=int, default=10000,
                       help='Maximum real windows to load (default: 10000)')
    parser.add_argument('--max_synth_windows', type=int, default=5000,
                       help='Maximum synthetic windows to analyze (default: 5000)')
    parser.add_argument('--distance_metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine'],
                       help='Distance metric to use')
    parser.add_argument('--check_synth_diversity', action='store_true',
                       help='Also compute synthetic-to-synthetic distances')
    parser.add_argument('--check_exact_matches', action='store_true',
                       help='Check for exact window matches (slower)')
    parser.add_argument('--output_dir', type=str, default='privacy_analysis',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')

    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load real data
    print(f"Loading real data from: {args.real_glob}")
    real_paths = sorted(glob.glob(args.real_glob))
    if not real_paths:
        raise FileNotFoundError(f"No files found matching: {args.real_glob}")

    print(f"Found {len(real_paths)} real data shards")

    real_windows_list = []
    for path in tqdm(real_paths, desc="Loading real shards"):
        windows = load_npz_windows(path)
        real_windows_list.append(windows)
        if sum(len(w) for w in real_windows_list) >= args.max_real_windows:
            break

    real_windows = np.concatenate(real_windows_list, axis=0)
    if len(real_windows) > args.max_real_windows:
        indices = np.random.choice(len(real_windows), args.max_real_windows, replace=False)
        real_windows = real_windows[indices]

    print(f"Loaded {len(real_windows)} real windows")

    # Load synthetic data
    print(f"\nLoading synthetic data from: {args.synth}")
    synth_windows = load_npz_windows(args.synth, max_windows=args.max_synth_windows)
    print(f"Loaded {len(synth_windows)} synthetic windows")

    # Compute embeddings
    print("\nComputing embeddings for real windows...")
    real_embeds = compute_embeddings_batch(real_windows)

    print("Computing embeddings for synthetic windows...")
    synth_embeds = compute_embeddings_batch(synth_windows)

    # Compute synthetic -> real distances
    nn_dist_synth_to_real = compute_nearest_neighbor_distances(
        synth_embeds, real_embeds,
        distance_metric=args.distance_metric
    )

    # Compute synthetic -> synthetic distances (optional)
    nn_dist_synth_to_synth = None
    if args.check_synth_diversity:
        print("\nComputing synthetic-to-synthetic distances for diversity analysis...")
        # For each synthetic sample, find nearest OTHER synthetic sample
        nn_dist_synth_to_synth = []
        for i in tqdm(range(len(synth_embeds))):
            # Exclude self by setting distance to inf
            distances = euclidean_distance_matrix(
                synth_embeds[i:i+1], synth_embeds
            )[0]
            distances[i] = np.inf
            nn_dist_synth_to_synth.append(np.min(distances))
        nn_dist_synth_to_synth = np.array(nn_dist_synth_to_synth)

    # Check for exact matches (optional)
    exact_matches = 0
    if args.check_exact_matches:
        exact_matches = compute_exact_matches(synth_windows, real_windows)

    # Compute statistics
    stats_synth_to_real = compute_statistics(nn_dist_synth_to_real)
    stats_synth_to_synth = compute_statistics(nn_dist_synth_to_synth) if nn_dist_synth_to_synth is not None else None

    # Generate plots
    plot_path_real = os.path.join(args.output_dir, 'nn_distances_synth_to_real.png')
    plot_distance_distribution(nn_dist_synth_to_real, plot_path_real,
                              title_suffix="(Synthetic -> Real)")

    if nn_dist_synth_to_synth is not None:
        plot_path_synth = os.path.join(args.output_dir, 'nn_distances_synth_to_synth.png')
        plot_distance_distribution(nn_dist_synth_to_synth, plot_path_synth,
                                  title_suffix="(Synthetic -> Synthetic)")

    # Save numerical results with UTF-8 encoding
    results_path = os.path.join(args.output_dir, 'privacy_analysis_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Privacy Analysis Results\n")
        f.write(f"{'='*70}\n")
        f.write(f"Real data pattern: {args.real_glob}\n")
        f.write(f"Synthetic data: {args.synth}\n")
        f.write(f"Distance metric: {args.distance_metric}\n")
        f.write(f"\nSynthetic -> Real Statistics:\n")
        for key, val in stats_synth_to_real.items():
            f.write(f"  {key}: {val:.6f}\n")
        if stats_synth_to_synth:
            f.write(f"\nSynthetic -> Synthetic Statistics:\n")
            for key, val in stats_synth_to_synth.items():
                f.write(f"  {key}: {val:.6f}\n")
        f.write(f"\nExact matches: {exact_matches}\n")

    print(f"\nSaved results to {results_path}")

    # Print analysis report
    print_analysis_report(stats_synth_to_real, stats_synth_to_synth,
                         exact_matches, len(synth_windows), len(real_windows))

    # Save distance arrays for further analysis
    np.savez(os.path.join(args.output_dir, 'nn_distances.npz'),
             synth_to_real=nn_dist_synth_to_real,
             synth_to_synth=nn_dist_synth_to_synth if nn_dist_synth_to_synth is not None else np.array([]))

    print(f"All results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

#python evaluation_metrics/privacy_analysis.py --real_glob "window_shards/compress-gzip/train/*.npz" --synth "art_outputs/generated_traces/ar/compress-gzip/synth_100k.npz" --max_real_windows 20000 --max_synth_windows 10000 --distance_metric euclidean --check_synth_diversity --check_exact_matches --output_dir evaluation_metrics/privacy_analysis_compress_gzip --seed 42

# to check for ALL 100K Synthetic Samples
#python evaluation_metrics/privacy_analysis.py --real_glob "window_shards/compress-gzip/train/*.npz" --synth "art_outputs/generated_traces/ar/compress-gzip/synth_100k.npz" --max_real_windows 50000 --max_synth_windows 100000 --distance_metric euclidean --check_synth_diversity --check_exact_matches --output_dir evaluation_metrics/privacy_analysis_full_100k --seed 42

#Option 2: Quick Check Without Exact Match Detection (Much Faster)
#python evaluation_metrics/privacy_analysis.py --real_glob "window_shards/compress-gzip/train/*.npz" --synth "art_outputs/generated_traces/ar/compress-gzip/synth_100k.npz" --max_real_windows 30000 --max_synth_windows 100000 --distance_metric euclidean --check_synth_diversity --output_dir evaluation_metrics/privacy_analysis_full_100k_fast --seed 42
