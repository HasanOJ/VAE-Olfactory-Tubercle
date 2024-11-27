# VAE-Olfactory-Tubercle

This repository provides a training pipeline for a Variational Autoencoder (VAE) to analyze brain tile data from olfactory tubercle microscopy images. The pipeline is designed to efficiently train and evaluate the VAE, leveraging density-based or random sampling for data preparation. 

The implementation is compatible with PyTorch Lightning and optimized for use with Google Colab.

---

## Quick Start

You can easily get started with the training pipeline in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hVIrS2AURs5P22nzMAcilHrTJD8VFGNg?usp=sharing)

### Steps:
1. Click the "Open in Colab" badge above or [this link](https://colab.research.google.com/drive/1hVIrS2AURs5P22nzMAcilHrTJD8VFGNg?usp=sharing).
2. Save a copy of the notebook to your Google Drive: **File → Save a copy in Drive**.
3. Execute the cells sequentially to prepare the data, train the model, and evaluate the results.

---

## Configuration

The default training configuration is as follows:
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
*Note: Adjust these parameters as needed to fit your specific requirements.*

---

## Sampling Strategies

Two sampling strategies are implemented to generate training tiles:
1. **Random Sampling (default):** Randomly selects tile locations without considering image properties.
2. **Density-Based Sampling:** Prioritizes tiles with higher structural content, computed as the inverse of mean pixel intensity.

The sampling strategy can be configured using the `--density` argument:
```bash
python main.py --density
```

---
