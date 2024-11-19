import numpy as np
import h5py
import torch
import random
from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Tuple
from torchvision import transforms
import pandas as pd

class BrainTileDataset(Dataset):
    """Dataset for accessing brain image tiles."""

    def __init__(self, file_path: str, global_stats: Dict[str, float], test_set: str, testing: bool = False, tile_size: int = 64, metadata: bool = False):
        """
        Initialize the dataset.
        
        Args:
            file_path: Path to the HDF5 file
            global_stats: Dictionary containing 'mean' and 'std' for standardization
            test_set: Brain ID to be used as the test set
            testing: Boolean flag to indicate if the dataset is for testing
            tile_size: Size of the tiles to extract
        """
        self.file_path = file_path
        self.global_mean = global_stats['mean']
        self.global_std = global_stats['std']
        self.images = {}
        self.testing = testing
        self.tile_size = tile_size
        self.metadata = metadata

        # Load images into RAM
        with h5py.File(self.file_path, 'r') as f:
            for brain in f.keys():
                if (testing and brain != test_set) or (not testing and brain == test_set):
                    continue
                self.images[brain] = {}
                for img_key in f[brain].keys():
                    self.images[brain][img_key] = torch.from_numpy(f[brain][img_key][()] / 255.0).float()
            self.images = pd.DataFrame([(brain, img_key, img_data) for brain, imgs in self.images.items() for img_key, img_data in imgs.items()],
                                       columns=['brain', 'image_key', 'image_data'])
                    

        # Define augmentations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5), # gaussian blur
            # transforms.RandomApply([transforms.RandomResizedCrop(size=tile_size, scale=(0.9, 1.0))], p=0.5),
            # transforms.RandomApply([transforms.ColorJitter(brightness=0.05, contrast=0.05)], p=0.5),
            # transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)], p=0.5),  # Gaussian noise
        ])

    
    def __getitem__(self, index) -> torch.Tensor:
        """
        Extract a tile from the specified location or return the entire image.
        
        Args:
            index: Tuple of (brain, image_key) or (brain, image_key, row, col, tile_size)
        
        Returns:
            Standardized tile or image as torch tensor
        """
        if len(index) == 1:
            brain = self.images.loc[index, 'brain']
            image_tensor = self.images.loc[index, 'image_data']
            image_tensor = (image_tensor - self.global_mean) / self.global_std
            return image_tensor, brain
        elif len(index) == 5:
            brain, image_key, row, col, tile_size = index
            
            # Load image data from RAM
            image_data = self.images.loc[(self.images['brain'] == brain) & (self.images['image_key'] == image_key), 'image_data'].values[0]
            tile = image_data[row:row + tile_size, col:col + tile_size]
            
            # Handle padding if needed
            if tile.shape != (tile_size, tile_size):
                padded_tile = torch.zeros((tile_size, tile_size))
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            # Convert to torch tensor, add channel dimension, normalize to 0-1 range, and standardize
            tile_tensor = tile.unsqueeze(0)
            tile_tensor = (tile_tensor - self.global_mean) / self.global_std
            
            # Apply augmentations if not in testing mode
            if not self.testing:
                tile_tensor = self.transform(tile_tensor)
            
            if self.metadata:
                return tile_tensor, (brain, image_key, row, col)
            return tile_tensor
        else:
            raise ValueError("Index must be a tuple of (brain, image_key) or (brain, image_key, row, col, tile_size)")

    def __len__(self) -> int:
        """Return total number of possible tiles across all images."""
        return len(self.images)


class DensityBasedSampler(Sampler):
    """Sampler for density-based sampling of tiles."""

    def __init__(self, dataset: BrainTileDataset, samples_per_epoch: int, inverse_density: bool = False, density: bool = True):
        """
        Initialize the sampler.
        
        Args:
            dataset: BrainTileDataset instance
            num_samples: Number of samples to draw per epoch
            inverse_density: Boolean flag to use inverse density for sampling
            density: Boolean flag to use density-based sampling or random sampling
        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.tile_size = dataset.tile_size
        self.train_brains = dataset.images['brain'].unique()
        self.density = density
        if self.density:
            self.weights = self.calculate_tile_densities(inverse_density)

        # Invert densities to prioritize darker tiles
        # self.weights = np.array([1.0 / density for density in self.weights.values()])
        # self.weights /= self.weights.sum()  # Normalize to create a probability distribution

        # Pre-calculate valid tile positions for each image
        self.valid_positions = self._calculate_valid_positions()

    def _calculate_valid_positions(self) -> Dict:
        """Calculate valid tile positions for each image."""
        positions = self.dataset.images['image_data'].apply(lambda x: (max(0, x.shape[0] - self.tile_size), max(0, x.shape[1] - self.tile_size)))
        return torch.tensor(positions.tolist(), dtype=torch.int)

    def calculate_tile_densities(self, inverse_density: bool = False) -> Dict:
        """Calculate the average pixel intensity for each image."""
        densities =  self.dataset.images['image_data'].apply(lambda x: x.mean())
        densities = torch.stack(densities.to_list())
        if inverse_density:
            return 1.0 / densities
        return 1.0 - densities

    def __iter__(self):
        """Yield random tile locations based on density."""
        for _ in range(self.samples_per_epoch):
            # Randomly select a random image index
            if self.density:
                index = random.choices(self.dataset.images.index, weights=self.weights)[0]
            else:
                index = random.choices(self.dataset.images.index)[0]
            
            # Sample random position
            max_row, max_col = self.valid_positions[index]
            row = random.randint(0, max_row)
            col = random.randint(0, max_col)
            brain = self.dataset.images.loc[index, 'brain']
            image_key = self.dataset.images.loc[index, 'image_key']
            
            yield (brain, image_key, row, col, self.tile_size)

    def __len__(self):
        """Return the number of samples."""
        return self.samples_per_epoch