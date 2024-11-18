import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import os
import torch

def calculate_statistics(data_path='cell_data.h5', test_set='B20', cache_file='statistics_cache.json'):
    # Load cached statistics if they exist
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            if test_set in cache:
                print(f"Loading cached statistics for test set {test_set}...")
                return cache[test_set]

    # Calculate statistics if not cached
    all_pixels = []
    with h5py.File(data_path, 'r') as f:
        for brain in f.keys():
            if brain == test_set:
                continue
            for img in f[brain]:
                data = f[brain][img][()]
                all_pixels.append(data.flatten())

    all_pixels = np.concatenate(all_pixels) / 255.0  # Normalize to 0-1 range
    # global_min = float(np.min(all_pixels))  # Convert to Python float for JSON compatibility
    # global_max = float(np.max(all_pixels))
    global_mean = float(np.mean(all_pixels))
    global_std = float(np.std(all_pixels))

    # Update the cache with new statistics
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    cache[test_set] = {
        # "min": global_min,
        # "max": global_max,
        "mean": global_mean,
        "std": global_std
    }

    with open(cache_file, 'w') as f:
        json.dump(cache, f)

    return cache[test_set]

def visualize_batch(batch, metadata, global_stats):
    """
    Visualize a batch of tiles with metadata.
    
    Args:
        batch: Tensor of shape (batch_size, 1, tile_size, tile_size)
        metadata: List of tuples (brain, image_key, tile_row, tile_col)
        global_stats: Dictionary containing 'mean' and 'std' for denormalization
    """
    batch_size = batch.shape[0]
    n_rows = int(np.ceil(np.sqrt(batch_size)))
    n_cols = int(np.ceil(batch_size / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.flatten()
    
    # Denormalize batch for visualization
    denorm_batch = denormalize(batch, global_stats['mean'], global_stats['std'])
    
    for i in range(batch_size):
        axes[i].imshow(denorm_batch[i, 0].numpy(), cmap='gray')
        axes[i].axis('off')
        brain, image_key, tile_row, tile_col = metadata[0][i], metadata[1][i], metadata[2][i].item(), metadata[3][i].item()
        axes[i].set_title(f"{brain}-{image_key}\n({tile_row}, {tile_col})", fontsize=8)
        
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
    data = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return data, metadata

def visualize_reconstruction(batch, reconstructed, metadata):
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))
    for i in range(8):
        # Original
        axes[0, i].imshow(batch[i].squeeze(0).cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        brain, image_key, tile_row, tile_col = metadata[0][i], metadata[1][i], metadata[2][i].item(), metadata[3][i].item()
        axes[0, i].set_title(f"{brain}-{image_key}\n({tile_row}, {tile_col})", fontsize=8)
        
        # Reconstructed
        recon_img = reconstructed[i].squeeze(0).detach().cpu().numpy()
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()