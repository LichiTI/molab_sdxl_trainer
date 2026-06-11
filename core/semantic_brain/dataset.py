
"""
Mixed Semantic Dataset Loader
Part of Lulynx Neuro-Link Architecture

Handles efficient loading of text data for semantic pre-alignment.
Supports mixing multiple data sources (e.g., Natural Language + Danbooru Tags)
with configurable sampling weights.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import random
from typing import List, Iterator, Optional
from pathlib import Path

class MixedTextDataset(IterableDataset):
    def __init__(
        self, 
        file_paths: List[str], 
        weights: Optional[List[float]] = None,
        batch_size: int = 1,
        shuffle_buffer_size: int = 1000
    ):
        """
        Initialize the Mixed Text Dataset.
        
        Args:
            file_paths: List of paths to text files (.txt).
            weights: Sampling weights for each file. Defaults to uniform.
            batch_size: (Not strictly used here if using DataLoader, but good for internal buffering)
            shuffle_buffer_size: Size of buffer for local shuffling.
        """
        self.file_paths = [Path(p) for p in file_paths]
        self.weights = weights if weights else [1.0] * len(file_paths)
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Verify files exist
        for p in self.file_paths:
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")
                
        # Basic validation
        if len(self.weights) != len(self.file_paths):
            raise ValueError("Number of weights must match number of file paths")
            
        # Normalize weights
        total_w = sum(self.weights)
        self.probs = [w / total_w for w in self.weights]

    def _file_generator(self, path: Path) -> Iterator[str]:
        """Yields lines from a file infinitely (loops)"""
        while True:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    text = line.strip()
                    if text:
                        yield text
            # If file ends, loop back (infinite dataset)

    def __iter__(self):
        """
        Yields mixed text samples.
        """
        # Create generators for each file
        generators = [self._file_generator(p) for p in self.file_paths]
        
        buffer = []
        
        while True:
            # Replenish buffer
            while len(buffer) < self.shuffle_buffer_size:
                # Select source based on weights
                chosen_idx = random.choices(range(len(generators)), weights=self.probs, k=1)[0]
                try:
                    text = next(generators[chosen_idx])
                    buffer.append(text)
                except StopIteration:
                    # Should not happen with infinite generator, but safety check
                    break
            
            if not buffer:
                break
                
            # Pop random item
            idx = random.randint(0, len(buffer) - 1)
            yield buffer.pop(idx)

def get_dataloader(
    file_paths: List[str], 
    weights: List[float], 
    batch_size: int, 
    num_workers: int = 0
) -> DataLoader:
    """Factory to get the PyTorch DataLoader"""
    dataset = MixedTextDataset(file_paths, weights)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
