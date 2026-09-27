"""
Train a DCGAN on Fashion-MNIST.

Usage:
    python src/train.py --epochs 20 --batch-size 128
    python src/train.py --epochs 20 --resume checkpoints/latest.pth

Designed to run on CPU. On a typical laptop CPU, expect roughly 1-3 minutes
per epoch depending on hardware. 15-25 epochs already produce recognizable
clothing silhouettes; 40+ epochs look noticeably sharper.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import Generator, Discriminator, weights_init
from utils import save_sample_grid, save_checkpoint, LossLogger

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
LOSS_CSV = os.path.join(OUTPUT_DIR, "losses.csv")


def get_dataloader(batch_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # scale to [-1, 1] to match Generator's Tanh output
        ]
    )
    dataset = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


def train(epochs: int, batch_size: int, lr: float, latent_dim: int, resume: str = None):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    dataloader = get_dataloader(batch_size)

    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    start_epoch = 0
    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optim_g.load_state_dict(checkpoint["optim_g_state_dict"])
        optim_d.load_state_dict(checkpoint["optim_d_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {resume} at epoch {start_epoch}")

    fixed_noise = torch.randn(64, latent_dim, device=device)  # fixed noise to visually track progress
    logger = LossLogger(LOSS_CSV)

    real_label, fake_label = 1.0, 0.0

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        running_loss_g, running_loss_d = 0.0, 0.0
        running_d_real_acc, running_d_fake_acc = 0.0, 0.0
        num_batches = 0

        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # ---------------------
            #  Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # ---------------------
            discriminator.zero_grad()

            labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real, labels)
            loss_d_real.backward()
            d_real_acc = (output_real > 0.5).float().mean().item()

            noise = torch.randn(b_size, latent_dim, device=device)
            fake_images = generator(noise)
            labels.fill_(fake_label)
            output_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(output_fake, labels)
            loss_d_fake.backward()
            d_fake_acc = (output_fake <= 0.5).float().mean().item()

            loss_d = loss_d_real + loss_d_fake
            optim_d.step()

            # ---------------------
            #  Train Generator: maximize log(D(G(z)))
            #  (non-saturating trick: minimize -log(D(G(z))) via real labels)
            # ---------------------
            generator.zero_grad()
            labels.fill_(real_label)
            output = discriminator(fake_images)
            loss_g = criterion(output, labels)
            loss_g.backward()
            optim_g.step()

            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item()
            running_d_real_acc += d_real_acc
            running_d_fake_acc += d_fake_acc
            num_batches += 1

        avg_loss_g = running_loss_g / num_batches
        avg_loss_d = running_loss_d / num_batches
        avg_d_real_acc = running_d_real_acc / num_batches
        avg_d_fake_acc = running_d_fake_acc / num_batches
        logger.log(epoch, avg_loss_g, avg_loss_d, avg_d_real_acc, avg_d_fake_acc)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss_G: {avg_loss_g:.4f} Loss_D: {avg_loss_d:.4f} "
            f"D(real): {avg_d_real_acc:.2f} D(fake): {avg_d_fake_acc:.2f} "
            f"({elapsed:.1f}s)"
        )

        with torch.no_grad():
            generator.eval()
            samples = generator(fixed_noise)
            generator.train()
        save_sample_grid(samples, os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1:03d}.png"))

        save_checkpoint(generator, discriminator, optim_g, optim_d, epoch, os.path.join(CHECKPOINT_DIR, "latest.pth"))
        save_checkpoint(generator, discriminator, optim_g, optim_d, epoch, os.path.join(CHECKPOINT_DIR, "generator.pth"))

    print("Training complete. Checkpoints saved in:", CHECKPOINT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DCGAN on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        resume=args.resume,
    )
