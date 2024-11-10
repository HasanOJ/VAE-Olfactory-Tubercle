import numpy as np
import h5py
import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple
from torchvision import transforms

class BrainTileDataset(Dataset):
    """Dataset for accessing brain image tiles."""

    def __init__(self, file_path: str, global_stats: Dict[str, float], test_set: str, testing: bool = False, tile_size: int = 64):
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
        self.global_mean = global_stats['mean'] / 255.0  # Normalize to 0-1 range
        self.global_std = global_stats['std'] / 255.0
        self.images = {}
        self.testing = testing
        self.tile_size = tile_size

        # Load images into RAM
        with h5py.File(self.file_path, 'r') as f:
            for brain in f.keys():
                if (testing and brain != test_set) or (not testing and brain == test_set):
                    continue
                self.images[brain] = {}
                for img_key in f[brain].keys():
                    self.images[brain][img_key] = f[brain][img_key][()]

        # Define augmentations
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5), # gaussian blur
            # transforms.RandomApply([transforms.RandomResizedCrop(size=tile_size, scale=(0.9, 1.0))], p=0.5),
            # transforms.RandomApply([transforms.ColorJitter(brightness=0.05, contrast=0.05)], p=0.5),
            # transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)], p=0.5),  # Gaussian noise
        ])

    
    def __getitem__(self, index: Tuple[str, str, int, int, int]) -> torch.Tensor:
        """
        Extract a tile from the specified location.
        
        Args:
            index: Tuple of (brain, image_key, row, col, tile_size)
        
        Returns:
            Standardized tile as torch tensor with shape (1, tile_size, tile_size)
        """
        brain, image_key, row, col, tile_size = index
        
        # Load image data from RAM
        image_data = self.images[brain][image_key]
        tile = image_data[row:row + tile_size, col:col + tile_size]
        
        # Handle padding if needed
        if tile.shape != (tile_size, tile_size):
            padded_tile = np.zeros((tile_size, tile_size))
            padded_tile[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded_tile
        
        # Convert to torch tensor, add channel dimension, normalize to 0-1 range, and standardize
        tile_tensor = torch.from_numpy(tile).float().unsqueeze(0) / 255.0
        tile_tensor = (tile_tensor - self.global_mean) / self.global_std
        
        # Apply augmentations if not in testing mode
        if not self.testing:
            tile_tensor = self.augmentations(tile_tensor)
        
        return tile_tensor

    def __len__(self) -> int:
        """Return total number of possible tiles across all images."""
        return sum(len(brain_images) for brain_images in self.images.values())


class RandomTileSampler(Sampler):
    """Sampler for randomly selecting tile locations."""

    def __init__(self, dataset: BrainTileDataset, 
                 samples_per_epoch: int, tile_size: int):
        """
        Initialize the sampler.
        
        Args:
            dataset: BrainTileDataset instance
            train_brains: List of brain IDs to sample from
            samples_per_epoch: Number of tiles to sample per epoch
            tile_size: Size of tiles to extract
        """
        self.dataset = dataset
        self.train_brains = list(dataset.images.keys())
        self.samples_per_epoch = samples_per_epoch
        self.tile_size = tile_size
        
        # Pre-calculate valid tile positions for each image
        self.valid_positions = self._calculate_valid_positions()

    def _calculate_valid_positions(self) -> Dict:
        """Calculate valid tile positions for each image."""
        positions = {}
        for brain in self.train_brains:
            positions[brain] = {}
            for img_key in self.dataset.images[brain].keys():
                shape = self.dataset.images[brain][img_key].shape
                max_row = max(0, shape[0] - self.tile_size)
                max_col = max(0, shape[1] - self.tile_size)
                positions[brain][img_key] = (max_row, max_col)
        return positions

    def __iter__(self):
        """Yield random tile locations."""
        for _ in range(self.samples_per_epoch):
            # Randomly select brain and image
            brain = random.choice(self.train_brains)
            image_key = random.choice(list(self.valid_positions[brain].keys()))
            
            # Get valid position ranges
            max_row, max_col = self.valid_positions[brain][image_key]
            
            # Sample random position
            row = random.randint(0, max_row)
            col = random.randint(0, max_col)
            
            yield (brain, image_key, row, col, self.tile_size)

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self.samples_per_epoch


def create_dataloader(file_path: str, global_stats: Dict[str, float], 
                      batch_size: int = 8, samples_per_epoch: int = 1024, **kwargs):
    """
    Create DataLoaders for training and testing.
    
    Args:
        file_path: Path to HDF5 file
        global_stats: Dictionary with global statistics
        test_set: Brain ID for testing
        tile_size: Size of tiles to extract
        batch_size: Number of tiles per batch
        samples_per_epoch: Number of tiles to sample per epoch
    """
    # Create training dataset
    train_dataset = BrainTileDataset(file_path, global_stats, test_set=kwargs['test_set'], tile_size=kwargs['tile_size'])
    
    # Create testing dataset
    # test_dataset = BrainTileDataset(file_path, global_stats, test_set=test_set, testing=True)
    
    # Create training sampler
    train_sampler = RandomTileSampler(train_dataset, samples_per_epoch, kwargs['tile_size'])
    
    # Create training data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda x: torch.stack([item for item in x])
    )
    
    # Create testing data loader without a sampler
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,  # No need for a sampler, use sequential access
    #     collate_fn=lambda x: torch.stack([item for item in x])
    # )
    
    # return train_dataloader, test_dataloader, train_dataset, test_dataset
    return train_dataloader, train_dataset