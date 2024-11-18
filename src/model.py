import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: list = None):
        super(Encoder, self).__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int, hidden_dims: list = None):
        super(Decoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)
        modules = []
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

class VAE(pl.LightningModule):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, latent_dim: int = 128, 
                 beta: float = 4, gamma: float = 1000, max_capacity: int = 25, 
                 Capacity_max_iter: int = 1e5, loss_type: str = 'H', hidden_dims: list = None, 
                 learning_rate: float = 1e-3, img_size: int = 64):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(out_channels, latent_dim, hidden_dims)
        self.beta = beta
        self.gamma = gamma
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.num_iter = 0
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def loss_function(self, x_hat, x, mu, log_var, kld_weight=1.0):
        self.num_iter += 1
        recons_loss = F.mse_loss(x_hat, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(x.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Invalid loss type. Choose either 'H' or 'B'.")

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
    
    def training_step(self, batch, batch_idx):
        x = batch  # Assuming dataset returns (image, label) but we only need images here
        x_hat, mu, log_var = self.forward(x)
        losses = self.loss_function(x_hat, x, mu, log_var, kld_weight=0.1)
        self.log_dict({'train_loss': losses['loss'], 'train_reconstruction_loss': losses['Reconstruction_Loss'], 'train_kld': losses['KLD']})
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, log_var = self.forward(x)
        losses = self.loss_function(x_hat, x, mu, log_var, kld_weight=0.1)
        self.log_dict({'val_loss': losses['loss'], 'val_reconstruction_loss': losses['Reconstruction_Loss'], 'val_kld': losses['KLD']})
        return losses['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def sample(self, num_samples: int, device: torch.device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decoder(z)

    def generate(self, x):
        return self.forward(x)[0]