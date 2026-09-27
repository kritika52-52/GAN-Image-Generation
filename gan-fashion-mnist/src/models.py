"""
DCGAN model definitions for Fashion-MNIST (28x28, 1-channel images).

Architecture follows the DCGAN paper (Radford et al., 2015) guidelines:
- Replace pooling with strided convolutions (Discriminator) and
  fractional-strided convolutions / transposed convolutions (Generator).
- Use BatchNorm in both G and D (except G's output layer and D's input layer).
- Remove fully connected hidden layers.
- ReLU activation in G for all layers except output (Tanh).
- LeakyReLU activation in D for all layers.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Maps a latent noise vector z (shape: [batch, latent_dim, 1, 1]) to a
    28x28x1 fake image in the range [-1, 1] (matches Tanh output + normalized data).
    """

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # Input: (latent_dim) x 1 x 1  ->  (feature_maps*4) x 7 x 7
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # (feature_maps*4) x 7 x 7  ->  (feature_maps*2) x 14 x 14
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # (feature_maps*2) x 14 x 14  ->  1 x 28 x 28
            nn.ConvTranspose2d(feature_maps * 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z arrives as [batch, latent_dim] -> reshape to [batch, latent_dim, 1, 1]
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """
    Maps a 28x28x1 image to a single scalar 'realness' probability.
    """

    def __init__(self, feature_maps: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            # 1 x 28 x 28  ->  (feature_maps) x 14 x 14
            nn.Conv2d(1, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps) x 14 x 14  ->  (feature_maps*2) x 7 x 7
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps*2) x 7 x 7  ->  1 x 1 x 1
            nn.Conv2d(feature_maps * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.net(img)
        return out.view(-1, 1).squeeze(1)


def weights_init(m: nn.Module) -> None:
    """DCGAN paper weight initialization: N(0, 0.02) for Conv/BatchNorm layers."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
