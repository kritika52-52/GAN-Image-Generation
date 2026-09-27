# Fashion-MNIST GAN — Image Generation with a Web UI

A **DCGAN (Deep Convolutional Generative Adversarial Network)** trained from scratch
on the Fashion-MNIST dataset, with a **Streamlit web UI** to generate new clothing
images, explore the latent space, and inspect training curves.

Runs entirely on **CPU** — no GPU required.

---

## 🗂️ Project Structure

```
gan-fashion-mnist/
├── app.py                 # Streamlit UI (run this to use the app)
├── requirements.txt
├── src/
│   ├── models.py           # Generator & Discriminator (DCGAN architecture)
│   ├── train.py             # Training script (run this first)
│   └── utils.py               # Checkpointing, image grid saving, loss logging
├── data/                    # Fashion-MNIST auto-downloads here (created on first run)
├── checkpoints/               # Saved model weights (created during training)
└── outputs/                    # Sample image grids + losses.csv (created during training)
```

## ⚙️ Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies (CPU-only PyTorch build recommended — smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 🏋️ Step 1: Train the model

```bash
python src/train.py --epochs 20 --batch-size 128
```

- The Fashion-MNIST dataset (~30MB) downloads automatically on first run.
- Each epoch takes roughly **1-3 minutes on a typical laptop CPU**.
- After every epoch, a sample image grid is saved to `outputs/epoch_XXX.png` so you
  can watch the model improve, and a checkpoint is saved to `checkpoints/generator.pth`.
- **Resume training** at any time:
  ```bash
  python src/train.py --epochs 40 --resume checkpoints/latest.pth
  ```
- Results in practice: ~15-20 epochs → recognizable clothing silhouettes;
  ~40+ epochs → noticeably sharper edges and shapes.

## 🖥️ Step 2: Launch the UI

```bash
streamlit run app.py
```

This opens a browser tab with four tabs:
| Tab | What it does |
|---|---|
| 🎲 Generate | Produce a grid of random generated clothing images |
| 🔀 Latent Interpolation | Morph smoothly between two random latent vectors |
| 📉 Training Curves | Plot generator/discriminator loss and accuracy over epochs |
| ℹ️ About | Architecture summary for quick reference (handy in interviews!) |

You can open the UI **before** training finishes — it will just show a "no
checkpoint found yet" message until `checkpoints/generator.pth` exists.

## 🧠 How it works (short version)

Two neural networks are trained in opposition:
- The **Generator** takes a random noise vector (100 numbers) and learns to turn it
  into a 28×28 image that looks like real clothing.
- The **Discriminator** looks at an image (real or generated) and learns to tell
  real Fashion-MNIST images apart from the Generator's fakes.

They're trained together: the Generator gets better at fooling the Discriminator,
and the Discriminator gets better at catching fakes — pushing both to improve until
the Generator produces convincing images. See `INTERVIEW_NOTES.md` for the full
theory, equations, and likely interview questions with answers.

## 🚀 Possible extensions (good talking points for interviews)

- **Conditional GAN (cGAN):** feed the class label into both G and D so you can
  choose *which* clothing category to generate.
- **WGAN-GP:** replace BCE loss with Wasserstein loss + gradient penalty for more
  stable training and less mode collapse.
- **FID score:** add a quantitative image-quality metric instead of only eyeballing
  sample grids.
- **Deploy the Streamlit app** (e.g. Streamlit Community Cloud / Hugging Face
  Spaces / Render) so recruiters can try it live from a link on your resume.

## 🔧 Troubleshooting

- **`ModuleNotFoundError: torch`** → activate your virtual environment before running commands.
- **Training very slow** → reduce `--batch-size` to 64, or reduce `--epochs`; the
  architecture is intentionally small (2-3 conv layers) to stay CPU-friendly.
- **Streamlit shows stale images** → the app caches the loaded model with
  `@st.cache_resource`; restart Streamlit after training finishes to pick up new
  weights, or add a "Reload model" button (see `app.py`) if you keep retraining
  during a session.
