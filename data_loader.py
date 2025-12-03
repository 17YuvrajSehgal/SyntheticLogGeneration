"""
Data loading and preprocessing module for kernel trace sequences.
Based on the ICSE 2025 paper methodology.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os


class TraceDataset(Dataset):
    """Dataset class for loading kernel trace sequences."""
    
    def __init__(self, data_path, sequence_length=200, normalize=True, min_val=None, max_val=None):
        """
        Args:
            data_path: Path to the training or testing data file
            sequence_length: Length of sequences to use
            normalize: Whether to normalize the data
            min_val: Minimum value for normalization (if None, computed from data)
            max_val: Maximum value for normalization (if None, computed from data)
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Load data from text file
        sequences = []
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    seq = [int(x) for x in line.split(',')]
                    if sequence_length is None or len(seq) >= sequence_length:
                        if sequence_length is not None:
                            sequences.append(seq[:sequence_length])
                        else:
                            sequences.append(seq)
                except ValueError as e:
                    print(f"Warning: Skipping invalid line: {line[:50]}... Error: {e}")
                    continue
        
        if len(sequences) == 0:
            raise ValueError(f"No valid sequences found in {data_path}. Check file format and sequence_length parameter.")
        
        self.data = np.array(sequences, dtype=np.float32)
        
        # Normalize to [0, 1] range if needed
        if normalize:
            self.min_val = min_val if min_val is not None else float(self.data.min())
            self.max_val = max_val if max_val is not None else float(self.data.max())
            if self.max_val > self.min_val:
                self.data = (self.data - self.min_val) / (self.max_val - self.min_val)
            else:
                self.data = np.zeros_like(self.data)
        else:
            self.min_val = float(self.data.min())
            self.max_val = float(self.data.max())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return torch.FloatTensor(self.data[idx])


def create_event_mapping(dataset_path, output_path=None):
    """
    Create a mapping from system calls to integer IDs based on frequency.
    This matches the preprocessing described in the ICSE 2025 paper.
    
    Args:
        dataset_path: Path to directory containing training data
        output_path: Optional path to save the mapping
    """
    all_events = []
    
    # Collect all events from training files
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    for root, dirs, files in os.walk(dataset_path):
        if 'training' in files:
            train_file = os.path.join(root, 'training')
            try:
                with open(train_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        try:
                            events = [int(x) for x in line.split(',')]
                            all_events.extend(events)
                        except ValueError as e:
                            print(f"Warning: Skipping invalid line in {train_file}: {e}")
                            continue
            except Exception as e:
                print(f"Warning: Could not read {train_file}: {e}")
                continue
    
    # Count frequencies and sort
    event_counts = Counter(all_events)
    sorted_events = sorted(event_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Create mapping: event -> ID (0-indexed, sorted by frequency)
    event_to_id = {event: idx for idx, (event, _) in enumerate(sorted_events)}
    id_to_event = {idx: event for event, idx in event_to_id.items()}
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'event_to_id': event_to_id,
                'id_to_event': id_to_event,
                'num_events': len(event_to_id)
            }, f, indent=2)
    
    return event_to_id, id_to_event


def get_dataloader(data_path, batch_size=32, sequence_length=200, 
                   shuffle=True, normalize=True):
    """
    Create a DataLoader for trace sequences.
    
    Args:
        data_path: Path to training or testing data file
        batch_size: Batch size for training
        sequence_length: Length of sequences
        shuffle: Whether to shuffle the data
        normalize: Whether to normalize the data
    """
    dataset = TraceDataset(data_path, sequence_length, normalize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Example usage
    data_path = "Datasets/compress-gzip/sequence_length_200/training"
    loader = get_dataloader(data_path, batch_size=16)
    
    print(f"Dataset size: {len(loader.dataset)}")
    sample = next(iter(loader))
    print(f"Sample batch shape: {sample.shape}")

