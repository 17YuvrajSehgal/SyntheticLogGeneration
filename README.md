# Synthetic Kernel Trace Generation

This project implements generative AI models for synthesizing kernel trace sequences, based on research from:

1. **ICSE 2025**: "Execution Trace Reconstruction Using Diffusion-Based Generative Models" - Adapts the SSSDS4 (Structured State Space Diffusion S4) model for unconditional generation
2. **Tracyn Paper**: "Generative Artificial Intelligence based Peripherals Trace Synthesizer" - Provides insights on trace generation approaches

## Overview

This implementation focuses on **generating** synthetic kernel traces (system call sequences) rather than reconstructing missing events. The best-performing model from the ICSE 2025 paper (SSSDS4) is adapted for unconditional generation.

## Features

- **SSSDS4 Model**: Diffusion-based generative model with S4 (Structured State Space) layers for long-term dependency modeling
- **Training Pipeline**: Complete training script with validation and checkpointing
- **Generation Pipeline**: Script to generate synthetic kernel traces
- **Evaluation Metrics**: Implements accuracy, perfect rate, and ROUGE-L metrics from the research papers

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SyntheticLogGeneration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The project expects data in the following format:
- Text files with comma-separated integer sequences
- Each integer represents a system call ID
- Sequences are preprocessed and mapped based on frequency (as described in the ICSE 2025 paper)

Example:
```
10,10,14,14,14,14,13,13,13,13,18,18,10,10,10,10,...
```

## Usage

### Training

Train the SSSDS4 model on a dataset:

```bash
python train.py \
    --dataset compress-gzip \
    --sequence_length 200 \
    --batch_size 32 \
    --epochs 100 \
    --lr 2e-4 \
    --d_model 128 \
    --num_layers 4 \
    --save_dir checkpoints/compress-gzip
```

Arguments:
- `--dataset`: Dataset name (e.g., compress-gzip, ffmpeg, iozone)
- `--sequence_length`: Length of sequences (default: 200)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 2e-4)
- `--d_model`: Model dimension (default: 128)
- `--num_layers`: Number of S4 layers (default: 4)
- `--device`: Device to use (default: cuda)

### Generation

Generate synthetic kernel traces using a trained model:

```bash
python generate.py \
    --checkpoint checkpoints/compress-gzip/best.pt \
    --num_samples 100 \
    --seq_len 200 \
    --output_dir generated_traces/compress-gzip \
    --output_format both
```

Arguments:
- `--checkpoint`: Path to trained model checkpoint
- `--num_samples`: Number of sequences to generate (default: 100)
- `--seq_len`: Length of generated sequences (default: 200)
- `--batch_size`: Batch size for generation (default: 16)
- `--output_dir`: Directory to save generated traces
- `--output_format`: Output format - 'text', 'npy', or 'both' (default: text)

### Evaluation

Evaluate generated traces against real traces:

```bash
python evaluate.py \
    --generated generated_traces/compress-gzip/generated_traces.txt \
    --real Datasets/compress-gzip/sequence_length_200/testing \
    --output evaluation_results.json
```

This will compute:
- **Accuracy**: Proportion of events correctly positioned
- **Perfect Rate**: Proportion of sequences perfectly matched
- **ROUGE-L**: Longest Common Subsequence-based score
- **Distribution Metrics**: KL divergence, unique events, etc.

## Model Architecture

The SSSDS4 model combines:
1. **Diffusion Process**: Iterative denoising from noise to data
2. **S4 Layers**: Structured State Space models for efficient long-term dependency modeling
3. **Timestep Embeddings**: Sinusoidal embeddings for diffusion timesteps

Key components:
- Input projection with timestep conditioning
- Multiple S4 blocks with residual connections
- Output projection to generate sequences

## Results

Based on the ICSE 2025 paper, the SSSDS4 model achieved:
- **Accuracy**: 81.62% (for 10-event reconstruction)
- **Perfect Rate**: 74.27%
- **ROUGE-L**: 90.84%

For generation tasks, results may vary. The model should be evaluated on your specific datasets.

## Project Structure

```
.
├── data_loader.py          # Data loading and preprocessing
├── train.py                # Training script
├── generate.py             # Generation script
├── evaluate.py             # Evaluation script
├── models/
│   ├── __init__.py
│   ├── s4.py              # S4 layer implementation
│   └── sssds4.py          # SSSDS4 model implementation
├── Datasets/              # Training and test datasets
├── checkpoints/           # Saved model checkpoints
├── generated_traces/      # Generated synthetic traces
└── requirements.txt       # Python dependencies
```

## Notes

1. **Normalization**: The current implementation normalizes data to [0, 1]. You may need to adjust the denormalization in `generate.py` based on your actual data statistics.

2. **S4 Implementation**: The S4 layer is a simplified version. For production use, consider using optimized S4 implementations from libraries like `s4-pytorch`.

3. **Generation Quality**: The model generates continuous values that are quantized to integers. Fine-tuning the quantization process may improve results.

4. **Computational Requirements**: Training requires GPU for reasonable performance. Generation can be done on CPU but will be slower.

## Future Improvements

- [ ] Implement full S4 architecture with optimized state space operations
- [ ] Add support for conditional generation (e.g., based on application type)
- [ ] Implement Tracyn-style calibration post-processing
- [ ] Add support for variable-length sequence generation
- [ ] Integrate additional evaluation metrics
- [ ] Support for multi-dimensional traces (with timestamps, arguments, etc.)

## Citation

If you use this code, please cite the original papers:

1. Janecek, M., Ezzati-Jivan, N., & Hamou-Lhadj, A. (2025). Execution Trace Reconstruction Using Diffusion-Based Generative Models. ICSE 2025.

2. Huang, Z., et al. (2024). Project Tracyn: Generative Artificial Intelligence based Peripherals Trace Synthesizer.

## License

[Add your license here]

## Contact

[Add contact information here]

