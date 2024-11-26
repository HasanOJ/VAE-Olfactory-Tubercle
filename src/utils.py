import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

def interpolate_tiles(model, sample_batch, metadata, global_mean, global_std, same_section=True):

    # Select two tiles within the same section (image_keys)
    image_keys = metadata[1]

    # Build a dataframe with images and their corresponding keys
    df = pd.DataFrame({'key': image_keys})

    if same_section:
        # Choose a key with more than one tile
        key1 = key2 = df['key'].value_counts().index[0]

        # Select two random tiles
        tile_1, tile_2 = sample_batch[df[df['key'] == key1].sample(2).index]
    else:
        # Select two random tiles from different sections
        while True:
            idx = df.sample(2).index
            if df.loc[idx[0], 'key'] != df.loc[idx[1], 'key']:
                break
        tile_1, tile_2 = sample_batch[idx]
        key1, key2 = df.loc[idx, 'key']

    # Plot the two tiles
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    axes[0].imshow(denormalize(tile_1.squeeze(0).cpu().numpy(), global_mean, global_std).squeeze(), cmap='gray')
    axes[0].axis('off')
    axes[0].set_title(f"Key: {key1}")
    axes[1].imshow(denormalize(tile_2.squeeze(0).cpu().numpy(), global_mean, global_std).squeeze(), cmap='gray')
    axes[1].axis('off')
    axes[1].set_title(f"Key: {key2}")
    plt.suptitle(f"Two Random Tiles from {'the Same Section' if same_section else 'Different Sections'}", fontsize=14)
    plt.show()

    z1, _ = model.encoder(tile_1.unsqueeze(0))  # Encode tile 1
    z2, _ = model.encoder(tile_2.unsqueeze(0))  # Encode tile 2

    # Interpolate between latent representations
    k = 10  # Number of interpolation steps
    interpolated_latents = torch.stack([z1 + t * (z2 - z1) for t in torch.linspace(0, 1, k)], dim=0)

    # Decode interpolated representations
    interpolated_images = model.decoder(interpolated_latents).detach().cpu()

    # Visualize the interpolation
    fig, axes = plt.subplots(1, k, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(interpolated_images[i].squeeze(0), cmap='gray')
        ax.axis('off')
    plt.suptitle("Interpolation Between Two Tiles in Latent Space", fontsize=14)
    plt.show()

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