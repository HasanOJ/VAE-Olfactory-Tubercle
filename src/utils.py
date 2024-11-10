import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import os
import torch

def calculate_statistics(cache_file='statistics_cache.json', **kwargs):
    # Load cached statistics if they exist
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            if kwargs['test_set'] in cache:
                print(f"Loading cached statistics for test set {kwargs['test_set']}...")
                return cache[kwargs['test_set']]

    # Calculate statistics if not cached
    training_brains = kwargs['brain_names'].copy()
    training_brains.remove(kwargs['test_set'])
    
    all_pixels = []
    with h5py.File(kwargs['data_path'], 'r') as f:
        for brain in training_brains:
            for img in f[brain]:
                data = f[brain][img][()]
                all_pixels.append(data.flatten())

    all_pixels = np.concatenate(all_pixels)
    global_min = float(np.min(all_pixels))  # Convert to Python float for JSON compatibility
    global_max = float(np.max(all_pixels))
    global_mean = float(np.mean(all_pixels))
    global_std = float(np.std(all_pixels))

    # Update the cache with new statistics
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    cache[kwargs['test_set']] = {
        "min": global_min,
        "max": global_max,
        "mean": global_mean,
        "std": global_std
    }

    with open(cache_file, 'w') as f:
        json.dump(cache, f)

    return cache[kwargs['test_set']]

def visualize_batch(batch, **kwargs):
    """
    Visualize a batch of tiles.
    
    Args:
        batch: Tensor of shape (batch_size, 1, tile_size, tile_size)
        dataset: BrainTileDataset instance
    """
    batch_size = batch.shape[0]
    n_rows = int(np.ceil(np.sqrt(batch_size)))
    n_cols = int(np.ceil(batch_size / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.flatten()
    
    # Denormalize batch for visualization
    denorm_batch = denormalize(batch, kwargs['global_mean'], kwargs['global_std'])
    
    for i in range(batch_size):
        axes[i].imshow(denorm_batch[i, 0].numpy(), cmap='gray')
        axes[i].axis('off')
        
    # Hide empty subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def denormalize(tensor, global_mean, global_std):
    """
    Convert standardized tensor back to original range for visualization.
    """
    denorm = tensor * global_std + global_mean
    return denorm

def collate_fn(batch):
    return torch.stack([item for item in batch])