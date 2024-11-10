import argparse
import h5py
from data_processing import BrainTileDataset, DensityBasedSampler, RandomTileSampler
from utils import visualize_batch, calculate_statistics
import torch
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Train a model with specified test set.')
    parser.add_argument('--test_set', type=str, choices=['B01', 'B02', 'B05', 'B07', 'B20'], default='B20', help='Test set to use')
    parser.add_argument('--data_path', type=str, default='cell_data.h5', help='Path to the HDF5 dataset')
    parser.add_argument('--tile_size', type=int, default=64, help='Size of the tiles to extract')
    args = parser.parse_args()

    kwargs = vars(args)

    # Get list of brain names
    with h5py.File(kwargs['data_path'], 'r') as f:
        kwargs['brain_names'] = list(f.keys())

    if kwargs['test_set'] not in kwargs['brain_names']:
        raise ValueError(f"Test set {kwargs['test_set']} not found in HDF5 file.")
    
    # Calculate global statistics
    global_stats = calculate_statistics(**kwargs)
    kwargs['global_mean'] = global_stats['mean']
    kwargs['global_std'] = global_stats['std']

    # Create dataloaders
    train_dataset = BrainTileDataset(kwargs['data_path'], global_stats, test_set=kwargs['test_set'], tile_size=kwargs['tile_size'])
    density_sampler = DensityBasedSampler(train_dataset, samples_per_epoch=1024)
    random_sampler = RandomTileSampler(train_dataset, samples_per_epoch=1024)
    
    density_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=density_sampler,
        collate_fn=lambda x: torch.stack([item for item in x])
    )

    random_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=random_sampler,
        collate_fn=lambda x: torch.stack([item for item in x])
    )
    
    # Visualize first batch of training data
    for batch in density_dataloader:
        visualize_batch(batch, **kwargs)
        break

    for batch in random_dataloader:
        visualize_batch(batch, **kwargs)
        break

if __name__ == "__main__":
    main()