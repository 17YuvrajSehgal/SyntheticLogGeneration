
# Privacy Analysis: Usage Guide

## Quick Start

### Basic Usage
```bash
python privacy_analysis.py \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --synth "generated_traces/autoregressive/compress-gzip/synth_100k_best.npz" \
  --max_real_windows 10000 \
  --max_synth_windows 5000 \
  --output_dir privacy_analysis_results
```

### Full Analysis (with all features)
```bash
python privacy_analysis.py \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --synth "generated_traces/autoregressive/compress-gzip/synth_100k_best.npz" \
  --max_real_windows 20000 \
  --max_synth_windows 10000 \
  --distance_metric euclidean \
  --check_synth_diversity \
  --check_exact_matches \
  --output_dir privacy_analysis_results
```

### Compare Different Distance Metrics
```bash
# Euclidean distance
python privacy_analysis.py \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --synth "generated_traces/autoregressive/compress-gzip/synth_100k.npz" \
  --distance_metric euclidean \
  --output_dir privacy_results_euclidean

# Cosine distance
python privacy_analysis.py \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --synth "generated_traces/autoregressive/compress-gzip/synth_100k.npz" \
  --distance_metric cosine \
  --output_dir privacy_results_cosine
```

## Command-Line Arguments

- `--real_glob`: Glob pattern for real training data (required)
- `--synth`: Path to synthetic NPZ file (required)
- `--max_real_windows`: Max real windows to load (default: 10000)
- `--max_synth_windows`: Max synthetic windows to analyze (default: 5000)
- `--distance_metric`: Distance metric - 'euclidean' or 'cosine' (default: euclidean)
- `--check_synth_diversity`: Enable synthetic-to-synthetic distance analysis
- `--check_exact_matches`: Check for exact duplicate windows (slower but important)
- `--output_dir`: Directory for output files (default: privacy_analysis)
- `--seed`: Random seed for reproducibility (default: 42)

## Output Files

The script generates the following outputs in the specified output directory:

1. **nn_distances_synth_to_real.png**: 
   - 4-panel visualization showing distribution, CDF, box plot, and log-scale histogram
   - Primary metric for detecting memorization

2. **nn_distances_synth_to_synth.png** (if --check_synth_diversity):
   - Same visualization for synthetic-to-synthetic distances
   - Shows diversity within generated samples

3. **privacy_analysis_results.txt**:
   - Detailed statistics in text format
   - All percentiles and threshold metrics

4. **nn_distances.npz**:
   - Raw distance arrays for custom analysis
   - Can be loaded with np.load() for further processing

## Interpreting Results

### Distance Thresholds

**Median Distance:**
- < 0.5: ⚠️ High risk - potential memorization
- 0.5 - 1.5: ⚡ Moderate - some novelty present
- > 1.5: ✓ Good - strong novelty, low memorization risk

**Percentage Very Close (< 0.1):**
- > 10%: ⚠️ Concerning - many samples too similar to training data
- 5-10%: ⚡ Monitor - borderline acceptable
- < 5%: ✓ Good - acceptable similarity levels

**Exact Matches:**
- Any exact matches indicate direct memorization
- Should be 0 for good privacy preservation

### Privacy vs. Utility Trade-off

- **High distances**: Better privacy, but may indicate mode collapse or poor quality
- **Low distances**: Better utility/realism, but higher privacy risk
- **Goal**: Find the sweet spot where distances are high enough to prevent memorization 
  but low enough to maintain utility

### Synthetic-to-Synthetic Analysis

- Low synthetic-to-synthetic distances indicate lack of diversity (mode collapse)
- Should be comparable to or higher than synthetic-to-real distances
- Very low min distances suggest redundant generation

## Example Workflows

### Workflow 1: Quick Privacy Check
```bash
# Fast check with moderate sample size
python privacy_analysis.py \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --synth "generated_traces/autoregressive/compress-gzip/synth_1k.npz" \
  --max_real_windows 5000 \
  --max_synth_windows 1000 \
  --output_dir quick_check
```

### Workflow 2: Comprehensive Analysis
```bash
# Full analysis with all features
python privacy_analysis.py \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --synth "generated_traces/autoregressive/compress-gzip/synth_100k.npz" \
  --max_real_windows 50000 \
  --max_synth_windows 20000 \
  --check_synth_diversity \
  --check_exact_matches \
  --distance_metric euclidean \
  --output_dir comprehensive_privacy
```

### Workflow 3: Compare Multiple Models
```bash
# Analyze multiple synthetic datasets
for temp in 1.0 1.1 1.2; do
  python privacy_analysis.py \
    --real_glob "window_shards/compress-gzip/train/*.npz" \
    --synth "generated_traces/compress-gzip/synth_temp${temp}.npz" \
    --max_real_windows 10000 \
    --max_synth_windows 5000 \
    --check_synth_diversity \
    --output_dir "privacy_temp_${temp}"
done
```

## Feature Engineering Details

The script computes embeddings using multiple feature types:

1. **Statistical features**: mean, std, median for each channel (event, dt, cpu)
2. **Event histogram**: 50-bin histogram of event type distribution
3. **Delta time histogram**: 20-bin histogram of timing patterns
4. **CPU histogram**: 4-bin distribution across CPUs
5. **Transition features**: Unique bigram count (sequential patterns)

These features capture both local and global patterns in the trace windows.

## Performance Tips

- Start with smaller `--max_real_windows` and `--max_synth_windows` for faster iteration
- Use `--check_exact_matches` only when needed (it's O(n×m) complexity)
- Cosine distance is more robust to scale differences but slower to compute
- Results are saved incrementally - interrupt and resume is not supported

## Recommended Settings for Your Dataset

Based on your compress-gzip traces:
```bash
python privacy_analysis.py \
  --real_glob "window_shards/compress-gzip/train/*.npz" \
  --synth "art_outputs/generated_traces/ar/compress-gzip/synth_100k.npz" \
  --max_real_windows 20000 \
  --max_synth_windows 10000 \
  --distance_metric euclidean \
  --check_synth_diversity \
  --check_exact_matches \
  --output_dir privacy_analysis_compress_gzip \
  --seed 42
```
