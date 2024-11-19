# VAE-Olfactory-Tubercle

This guide explains how to run the Variational Autoencoder (VAE) training pipeline for brain tile data using Google Colab. The implementation is based on the [VAE-Olfactory-Tubercle repository](https://github.com/HasanOJ/VAE-Olfactory-Tubercle).

## Configuration

The default configuration is:
```python
config = {
    'img_channels': 1,
    'feature_dim': 128,
    'latent_dim': 128,
    'batch_size': 64,
    'learning_rate': 0.001,
    'max_epochs': 100,
    'test_set': 'B20',
    'data_path': 'cell_data.h5',
    'samples_per_epoch': 1024,
    'tile_size': 64
}
```

You can modify these parameters based on your needs.
> **NOTE:** As of now, `tile_size` can only be 64 because of the current model architecture .

## Training Pipeline

The training pipeline consists of several steps:

1. Data Preparation
   - Loads the dataset
   - Calculates global statistics
   - Creates data loaders

2. Model Setup
   - Initializes the VAE model
   - Sets up logging and callbacks
   - Configures the PyTorch Lightning trainer

3. Training
   - Trains the model with early stopping
   - Saves the best checkpoint

## Common Issues and Solutions

1. **Out of Memory Errors**
   - Reduce batch size in config
   - Reduce number of workers in DataLoader
   - Use mixed precision training (enabled by default)

2. **Slow Training**
   - Verify GPU is being used
   - Adjust number of workers in DataLoader
   - Consider reducing samples_per_epoch

3. **Data Loading Issues**
   - Verify file downloads completed successfully
   - Check paths are correct
   - Ensure all required files are present

## Additional Resources

- Original Repository: [VAE-Olfactory-Tubercle](https://github.com/HasanOJ/VAE-Olfactory-Tubercle)
- PyTorch Lightning Documentation: [Link](https://pytorch-lightning.readthedocs.io/)
- Google Colab Guide: [Link](https://colab.research.google.com/)
