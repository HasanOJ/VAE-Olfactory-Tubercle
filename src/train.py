import argparse
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from model import VAE
from data_processing import BrainTileDataset, DensityBasedSampler
from utils import calculate_statistics
import torch
from torch.utils.data import DataLoader
import h5py

torch.set_float32_matmul_precision('medium') # Use float32 matrix multiplication for better performance

def main():
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder")
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument('--test_set', type=str, choices=['B01', 'B02', 'B05', 'B07', 'B20'], default='B20', help='Test set to use')
    parser.add_argument('--data_path', type=str, default='cell_data.h5', help='Path to the HDF5 dataset')
    parser.add_argument("--samples_per_epoch", type=int, default=4096, help="Number of samples per epoch")
    parser.add_argument('--tile_size', type=int, default=128, help='Size of the tiles to extract')
    parser.add_argument('--density', action='store_true', help='Use density-based sampling', default=False)
    # parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Use seed_everything from PyTorch Lightning as an additional layer of seed setting
    # seed_everything(args.seed, workers=True)
    # torch.cuda.manual_seed(args.seed)
    # # Set a fixed value for CuDNN backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    kwargs = vars(args)

    # Get list of brain names
    with h5py.File(kwargs['data_path'], 'r') as f:
        kwargs['brain_names'] = list(f.keys())

    if kwargs['test_set'] not in kwargs['brain_names']:
        raise ValueError(f"Test set {kwargs['test_set']} not found in HDF5 file.")
    
    # Calculate global statistics
    global_stats = calculate_statistics(kwargs['data_path'], kwargs['test_set'])
    kwargs['global_mean'] = global_stats['mean']
    kwargs['global_std'] = global_stats['std']

    model = VAE(
        in_channels=kwargs['img_channels'],
        out_channels=kwargs['img_channels'],  # Assuming out_channels is the same as img_channels
        latent_dim=kwargs['latent_dim'],
        img_size=kwargs['tile_size'],
        learning_rate=kwargs['learning_rate'],
        beta=0.4,  # beta < 1 to increase the importance of reconstruction loss
        hidden_dims=None  # Default value, you can change this if needed
    )
    model.train()

    filename = f"{kwargs['test_set']}-{kwargs['tile_size']}"
    if kwargs['density']:
        filename += "-density"
    else:
        filename += "-random"

    # Create dataloaders
    train_dataset = BrainTileDataset(kwargs['data_path'], global_stats, test_set=kwargs['test_set'],
                                      tile_size=kwargs['tile_size'])
    test_dataset = BrainTileDataset(kwargs['data_path'], global_stats, test_set=kwargs['test_set'],
                                     tile_size=kwargs['tile_size'], testing=True)
    density_sampler = DensityBasedSampler(train_dataset, samples_per_epoch=kwargs['samples_per_epoch'], density=kwargs['density'])
    random_sampler = DensityBasedSampler(test_dataset, samples_per_epoch=kwargs['samples_per_epoch'], density=kwargs['density'])
    
    density_dataloader = DataLoader(
        train_dataset,
        batch_size=kwargs['batch_size'],
        sampler=density_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=kwargs['batch_size'],
        sampler=random_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True 
    )

    logger = TensorBoardLogger("tb_logs", name="vae-" + filename)

    # Define Model Checkpointing and Early Stopping callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint-' + filename,
        save_top_k=1,
        mode='min',
        enable_version_counter=False
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=7,  # Stop training if val_loss doesn't improve for some epochs
        mode='min',
        check_on_train_epoch_end=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=kwargs['max_epochs'],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1 if torch.cuda.is_available() else "auto",
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
        # log_every_n_steps=0,  # Disable step logging
        log_every_n_steps=kwargs['samples_per_epoch'] // kwargs['batch_size']  # Log once per epoch
    )
    ckpt_path = os.path.join('checkpoints', f'best-checkpoint-{filename}.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    trainer.fit(model, density_dataloader, test_dataloader, ckpt_path=ckpt_path)

    # Load the best checkpoint and initialize the model with saved weights
    # best_model_path = checkpoint_callback.best_model_path
    # trained_model = VAE.load_from_checkpoint(best_model_path)

if __name__ == "__main__":
    main()