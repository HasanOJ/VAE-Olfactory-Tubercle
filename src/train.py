import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import VAE
from data_processing import BrainTileDataset, DensityBasedSampler, RandomTileSampler
from utils import calculate_statistics, collate_fn
import torch
from torch.utils.data import DataLoader
import h5py

torch.set_float32_matmul_precision('medium') # Use float32 matrix multiplication for better performance

def main():
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder")
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=100)
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
    # random_sampler = RandomTileSampler(train_dataset, samples_per_epoch=1024)
    
    density_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=density_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # random_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=8,
    #     sampler=random_sampler,
    #     collate_fn=lambda x: torch.stack([item for item in x])
    # )

    model = VAE(
        in_channels=kwargs['img_channels'],
        out_channels=kwargs['img_channels'],  # Assuming out_channels is the same as img_channels
        latent_dim=kwargs['latent_dim'],
        img_size=kwargs['tile_size'],
        beta=4,  # Default value
        gamma=1000,  # Default value
        max_capacity=25,  # Default value
        Capacity_max_iter=1e5,  # Default value
        loss_type='H',  # Default value
        hidden_dims=None  # Default value, you can change this if needed
    )

    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=1, mode="min")
    logger = TensorBoardLogger("tb_logs", name="vae_experiment")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else None,
    )
    trainer.fit(model, density_dataloader)

if __name__ == "__main__":
    main()