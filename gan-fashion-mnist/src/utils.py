"""Helper utilities used by train.py and app.py"""

import csv
import os

import torch
from torchvision.utils import make_grid, save_image


def save_sample_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """
    Save a grid of generated images to disk.
    `images` is expected in range [-1, 1] (Tanh output); this rescales to [0, 1].
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, path)


def save_checkpoint(generator, discriminator, optim_g, optim_d, epoch: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optim_g_state_dict": optim_g.state_dict(),
            "optim_d_state_dict": optim_d.state_dict(),
        },
        path,
    )


def load_generator_checkpoint(generator, path: str, map_location="cpu") -> int:
    """Loads only the generator weights (used by the inference UI). Returns saved epoch number."""
    checkpoint = torch.load(path, map_location=map_location)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    return checkpoint.get("epoch", -1)


class LossLogger:
    """Appends per-epoch loss values to a CSV file so the UI can plot training curves."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss_g", "loss_d", "d_real_acc", "d_fake_acc"])

    def log(self, epoch: int, loss_g: float, loss_d: float, d_real_acc: float, d_fake_acc: float) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss_g, loss_d, d_real_acc, d_fake_acc])
