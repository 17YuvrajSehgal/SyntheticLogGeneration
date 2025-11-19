# Quick Start Guide

This guide will help you get started with generating synthetic kernel traces.

## Prerequisites

1. Python 3.8 or higher
2. CUDA-capable GPU (recommended for training)
3. PyTorch installed

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Analyze Your Dataset

First, understand your data:

```bash
python utils/data_stats.py --data_path Datasets/compress-gzip/sequence_length_200/training
```

This will show you:
- Number of sequences
- Sequence lengths
- Event frequency distribution
- Value ranges

### 2. Train a Model

Train the SSSDS4 model on your dataset:

```bash
python train.py --dataset compress-gzip --sequence_length 200 --batch_size 32 --epochs 100 --lr 2e-4 --save_dir checkpoints/compress-gzip

```

**Note**: Training can take several hours depending on your GPU and dataset size. For testing, you can reduce epochs to 10-20.

### 3. Generate Synthetic Traces

Once training is complete, generate new traces:

```bash
python generate.py --checkpoint checkpoints/compress-gzip/best.pt --num_samples 100 --seq_len 200 --output_dir generated_traces/compress-gzip
```

### 4. Evaluate Results

Compare generated traces with real ones:

```bash
python evaluate.py --generated generated_traces/compress-gzip/generated_traces.txt --real Datasets/compress-gzip/sequence_length_200/testing --output evaluation_results.json
```

## Example Pipeline

Run the complete pipeline with one command:

```bash
python example_usage.py
```

This will guide you through all steps interactively.

## Understanding the Output

### Generated Traces

The generated traces are saved as:
- `generated_traces.txt`: Text format (comma-separated integers)
- `generated_traces.npy`: NumPy array format
- `metadata.json`: Information about generation parameters

### Evaluation Metrics

The evaluation script computes:
- **Accuracy**: How many events match exactly (0-1)
- **ROUGE-L**: Longest common subsequence score (0-1)
- **Perfect Rate**: Percentage of perfectly matched sequences
- **Distribution Metrics**: KL divergence, unique events, etc.

## Tips for Better Results

1. **Data Quality**: Ensure your training data is clean and representative
2. **Hyperparameters**: 
   - Increase `d_model` for more capacity (128-256)
   - Increase `num_layers` for deeper models (4-8)
   - Adjust learning rate based on convergence
3. **Training Time**: 
   - Start with fewer epochs to test
   - Monitor validation loss
   - Use early stopping if loss plateaus
4. **Generation**:
   - Generate multiple samples and evaluate
   - Adjust quantization if values are out of range
   - Check normalization values match training data

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `sequence_length`
- Reduce `d_model` or `num_layers`

### Poor Generation Quality
- Train for more epochs
- Check data normalization
- Verify checkpoint loaded correctly
- Try different hyperparameters

### Normalization Issues
- The model saves normalization values in checkpoints
- If missing, check the dataset statistics
- Adjust min/max values in `generate.py` if needed

## Next Steps

1. Experiment with different datasets
2. Try different hyperparameters
3. Implement improvements from the research papers
4. Add conditional generation (e.g., by application type)
5. Extend to multi-dimensional traces

## Getting Help

- Check the README.md for detailed documentation
- Review the code comments for implementation details
- Refer to the original research papers for methodology

