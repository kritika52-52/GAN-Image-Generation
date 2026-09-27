"""
Streamlit UI for the Fashion-MNIST DCGAN.

Run with:
    streamlit run app.py

Features:
    - Generate a grid of random fashion images from the trained Generator
    - Latent-space interpolation between two random points (shows the model
      has learned a smooth, meaningful latent space rather than memorizing)
    - Training loss curve viewer (reads outputs/losses.csv written by train.py)
    - Model / architecture summary panel for quick reference
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torchvision.utils import make_grid
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from models import Generator

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "generator.pth")
LOSS_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "losses.csv")
LATENT_DIM = 100
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]  # Fashion-MNIST classes (shown for context only — this GAN is unconditional)

st.set_page_config(page_title="Fashion-MNIST GAN", page_icon="👗", layout="wide")


@st.cache_resource
def load_generator():
    """Loads the trained Generator once and caches it across reruns."""
    generator = Generator(latent_dim=LATENT_DIM)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        generator.load_state_dict(checkpoint["generator_state_dict"])
        epoch = checkpoint.get("epoch", -1)
    else:
        epoch = None
    generator.eval()
    return generator, epoch


def tensor_grid_to_pil(images: torch.Tensor, nrow: int) -> Image.Image:
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)


def main():
    st.title("👗 Fashion-MNIST GAN — Image Generator")
    st.caption(
        "A DCGAN (Deep Convolutional GAN) trained from scratch on the Fashion-MNIST dataset. "
        "Use the sidebar to generate new clothing images or explore the latent space."
    )

    generator, trained_epoch = load_generator()

    if not os.path.exists(CHECKPOINT_PATH):
        st.warning(
            "⚠️ No trained model found yet. Train it first from a terminal:\n\n"
            "`python src/train.py --epochs 20`\n\n"
            "The UI below will still run once `checkpoints/generator.pth` exists — "
            "just refresh the page after training."
        )
    else:
        st.success(f"Loaded generator checkpoint (trained through epoch {trained_epoch + 1}).")

    tab_generate, tab_interpolate, tab_training, tab_about = st.tabs(
        ["🎲 Generate", "🔀 Latent Interpolation", "📉 Training Curves", "ℹ️ About the Model"]
    )

    # ---------------- Tab 1: Random generation ----------------
    with tab_generate:
        st.subheader("Generate random samples")
        col1, col2 = st.columns([1, 3])
        with col1:
            num_images = st.slider("Number of images", 1, 64, 16, step=1)
            seed = st.number_input("Random seed", value=42, step=1)
            nrow = st.slider("Images per row", 1, 8, 4)
            generate_btn = st.button("Generate images", type="primary", use_container_width=True)

        with col2:
            if generate_btn or True:  # always show a grid, refresh on button click
                torch.manual_seed(int(seed) if generate_btn else np.random.randint(0, 100000))
                noise = torch.randn(num_images, LATENT_DIM)
                with torch.no_grad():
                    fake_images = generator(noise)
                grid_img = tensor_grid_to_pil(fake_images, nrow=nrow)
                st.image(grid_img, caption=f"{num_images} generated images", use_container_width=True)

    # ---------------- Tab 2: Latent interpolation ----------------
    with tab_interpolate:
        st.subheader("Latent space interpolation")
        st.write(
            "Smoothly morphs between two random latent vectors. If the model has learned "
            "a meaningful latent space (rather than memorizing), the intermediate images "
            "should look like plausible in-between clothing items, not random noise."
        )
        steps = st.slider("Number of interpolation steps", 4, 16, 8)
        interp_seed = st.number_input("Interpolation seed", value=7, step=1, key="interp_seed")

        if st.button("Interpolate", type="primary"):
            torch.manual_seed(int(interp_seed))
            z1 = torch.randn(1, LATENT_DIM)
            z2 = torch.randn(1, LATENT_DIM)
            alphas = torch.linspace(0, 1, steps).view(-1, 1)
            z_interp = z1 * (1 - alphas) + z2 * alphas
            with torch.no_grad():
                interp_images = generator(z_interp)
            grid_img = tensor_grid_to_pil(interp_images, nrow=steps)
            st.image(grid_img, caption="Interpolation from z1 to z2", use_container_width=True)

    # ---------------- Tab 3: Training curves ----------------
    with tab_training:
        st.subheader("Training loss history")
        if os.path.exists(LOSS_CSV_PATH):
            df = pd.read_csv(LOSS_CSV_PATH)
            st.line_chart(df.set_index("epoch")[["loss_g", "loss_d"]])
            st.caption(
                "Loss_G and Loss_D should roughly balance out. If Loss_D collapses to ~0, "
                "the discriminator is overpowering the generator — common GAN training failure mode."
            )
            st.line_chart(df.set_index("epoch")[["d_real_acc", "d_fake_acc"]])
            st.caption(
                "Discriminator accuracy on real vs fake batches — ideally both hover "
                "around 50-70% at equilibrium, not near 100%."
            )
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.info("No training log found yet. Run `python src/train.py` to generate `outputs/losses.csv`.")

    # ---------------- Tab 4: About ----------------
    with tab_about:
        st.subheader("Architecture summary")
        st.markdown(
            f"""
            **Type:** DCGAN (Deep Convolutional Generative Adversarial Network)
            **Dataset:** Fashion-MNIST (60,000 28×28 grayscale training images across 10 classes)
            **Latent dimension:** {LATENT_DIM}
            **Generator:** 3 transposed-conv layers, BatchNorm + ReLU, Tanh output
            **Discriminator:** 3 conv layers, BatchNorm + LeakyReLU(0.2), Sigmoid output
            **Loss:** Binary Cross-Entropy (standard non-saturating GAN loss)
            **Optimizer:** Adam (lr=0.0002, β1=0.5, β2=0.999) — standard DCGAN settings

            **Note:** This is an *unconditional* GAN — it generates random clothing items
            without letting you choose the class. The 10 Fashion-MNIST classes
            ({", ".join(CLASS_NAMES)}) are shown here only for reference; a natural
            extension is a **Conditional GAN (cGAN)** that lets you pick the class.
            """
        )


if __name__ == "__main__":
    main()
