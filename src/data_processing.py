import json
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class StatisticsCalculator:
    def __init__(self, h5_file_path='cell_data.h5', cache_file='statistics_cache.json'):
        self.h5_file_path = h5_file_path
        self.cache_file = cache_file

    def calculate_statistics(self, test_set):
        # Load cached statistics if they exist
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
                if test_set in cache:
                    print(f"Loading cached statistics for test set {test_set}...")
                    stats = cache[test_set]
                    return stats["min"], stats["max"], stats["mean"], stats["std"]

        # Calculate statistics if not cached
        training_brains = ['B01', 'B02', 'B05', 'B07', 'B20']
        training_brains.remove(test_set)
        
        all_pixels = []
        with h5py.File(self.h5_file_path, 'r') as f:
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
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

        cache[test_set] = {
            "min": global_min,
            "max": global_max,
            "mean": global_mean,
            "std": global_std
        }

        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)

        return global_min, global_max, global_mean, global_std

class MicroscopyTileDataset(Dataset):
    def __init__(self, h5_file_path, brains, global_mean, global_std, tile_size=64, preload_images=False):
        self.h5_file_path = h5_file_path
        self.brains = brains  # List of brain group names to load
        self.global_mean = global_mean
        self.global_std = global_std
        self.tile_size = tile_size
        self.images = {}  # Dictionary to hold images if preloaded
        if preload_images:
            self._preload_images()  # Load images to RAM for faster access

    def _preload_images(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            for brain in self.brains:
                self.images[brain] = {img: f[brain][img][()] for img in f[brain]}

    def __getitem__(self, index):
        # Unpack the tuple: (brain, image, row, column, tile_size)
        brain, image, row, col = index
        tile_size = self.tile_size

        # Retrieve image data either from RAM or from file
        if brain in self.images:
            img = self.images[brain][image]
        else:
            with h5py.File(self.h5_file_path, 'r') as f:
                img = f[brain][image][()]

        # Extract the tile from the image
        tile = img[row:row+tile_size, col:col+tile_size]

        # Ensure tile is the correct shape and pad if necessary
        if tile.shape != (tile_size, tile_size):
            tile = np.pad(tile, ((0, max(0, tile_size - tile.shape[0])),
                                 (0, max(0, tile_size - tile.shape[1]))),
                          mode='constant')

        # Convert tile to tensor, add channel dimension, and standardize
        tile_tensor = torch.tensor(tile, dtype=torch.float32).unsqueeze(0)
        tile_tensor = (tile_tensor - self.global_mean) / self.global_std

        return tile_tensor

    def __len__(self):
        # Define length as number of images * possible tile positions, or use a fixed count for epochs
        return 1000  # Example, adjust based on requirements

import random
from torch.utils.data import Sampler

class RandomTileSampler(Sampler):
    def __init__(self, h5_file_path, brains, tile_size=64, num_samples=1000):
        self.h5_file_path = h5_file_path
        self.brains = brains
        self.tile_size = tile_size
        self.num_samples = num_samples
        self.image_shapes = {}

        # Load the shapes of each image
        with h5py.File(self.h5_file_path, 'r') as f:
            for brain in self.brains:
                self.image_shapes[brain] = {img: f[brain][img].shape for img in f[brain]}

    def __iter__(self):
        for _ in range(self.num_samples):
            brain = random.choice(self.brains)
            image = random.choice(list(self.image_shapes[brain].keys()))
            img_shape = self.image_shapes[brain][image]
            
            # Randomly select a row and column within bounds
            row = random.randint(0, img_shape[0] - self.tile_size)
            col = random.randint(0, img_shape[1] - self.tile_size)
            
            yield (brain, image, row, col)

    def __len__(self):
        return self.num_samples  # Defines the number of samples per epoch


# from torch.utils.data import DataLoader

# # Instantiate the dataset and sampler
# dataset = MicroscopyTileDataset(h5_file_path='cell_data.h5',
#                                 brains=['B01', 'B02'],  # Example brains
#                                 global_mean=global_mean,
#                                 global_std=global_std,
#                                 tile_size=64,
#                                 preload_images=True)

# sampler = RandomTileSampler(h5_file_path='cell_data.h5',
#                             brains=['B01', 'B02'],
#                             tile_size=64,
#                             num_samples=1000)

# # Create DataLoader
# dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

# # Visualize a batch
# import matplotlib.pyplot as plt

# def visualize_batch(batch):
#     grid = plt.figure(figsize=(10, 10))
#     for i in range(len(batch)):
#         ax = grid.add_subplot(4, 4, i+1, xticks=[], yticks=[])
#         img = batch[i].squeeze(0) * global_std + global_mean  # Scale back for visualization
#         ax.imshow(img, cmap='gray')

# # Load a batch and visualize
# batch = next(iter(dataloader))
# visualize_batch(batch)
# plt.show()