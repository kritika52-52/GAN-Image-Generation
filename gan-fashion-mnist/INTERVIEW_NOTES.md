# Interview Notes: Fashion-MNIST DCGAN Project

Use this as a script to explain your project confidently, plus a Q&A bank for
common follow-up questions.

---

## 1. The 60-second pitch (elevator version)

"I built a GAN — a Generative Adversarial Network — that generates new images of
clothing items from scratch. It's trained on Fashion-MNIST, and it works by
having two neural networks compete: a **Generator** that tries to create
realistic images from random noise, and a **Discriminator** that tries to tell
real images apart from the Generator's fakes. As they train against each other,
the Generator gets good enough to fool the Discriminator, which means it's
learned to produce convincing clothing images. I also built a Streamlit UI on
top of it so you can generate images interactively, interpolate between two
random points in the latent space, and see the training loss curves — the
whole thing runs on CPU without needing a GPU."

---

## 2. Core theory: how does a GAN actually work?

A GAN has two networks trained in a **minimax game**:

- **Generator G(z):** takes random noise `z` (sampled from a simple distribution,
  e.g. Gaussian) and maps it to a fake image.
- **Discriminator D(x):** takes an image `x` and outputs the probability it's real.

They optimize the objective:

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

- D wants to **maximize** this: correctly classify real images as real (high
  `D(x)`) and fake images as fake (low `D(G(z))`).
- G wants to **minimize** this: make `D(G(z))` as close to 1 as possible, i.e.
  fool D into thinking fakes are real.

In practice, this project uses the standard **non-saturating trick**: instead of
literally minimizing `log(1 - D(G(z)))` (which has vanishing gradients early in
training), the Generator is trained to *maximize* `log(D(G(z)))` by using real
labels when computing its loss. This is implemented in `src/train.py`.

At convergence (in theory), the Generator's output distribution matches the real
data distribution, and the Discriminator can't do better than random guessing
(50% accuracy).

---

## 3. Why DCGAN specifically (architecture choices)

DCGAN (Radford et al., 2015) is the "recipe" that made GANs stable enough to
actually train reliably with convolutional networks. Key choices, and why:

| Choice | Why |
|---|---|
| Strided convolutions instead of pooling | Lets the network learn its own downsampling/upsampling instead of using a fixed, non-learnable operation |
| BatchNorm in G and D | Stabilizes training by keeping activations well-scaled; helps gradient flow in a deep-ish conv stack |
| No BatchNorm on D's input layer or G's output layer | Applying BatchNorm directly on real data / final output can introduce oscillation artifacts |
| ReLU in G, LeakyReLU(0.2) in D | LeakyReLU avoids the "dying ReLU" problem in the Discriminator, which is important because a dead Discriminator gives no useful gradient to G |
| Tanh output in G + normalizing real images to [-1, 1] | Keeps the Generator's output range matched to the real data range, avoids the Generator saturating |
| Adam optimizer, lr=0.0002, β1=0.5 | The DCGAN paper's tuned defaults — β1=0.5 (instead of the usual 0.9) reduces training oscillation |

## 4. Why Fashion-MNIST for this project

- Small (28×28 grayscale), so training is feasible on CPU in minutes-per-epoch,
  not hours.
- More visually interesting than plain MNIST digits (10 clothing categories)
  while still being a well-understood, standard benchmark — reviewers/interviewers
  recognize it immediately.
- Good sandbox to demonstrate GAN fundamentals before scaling to something
  heavier (CelebA faces, custom datasets) if there's more compute available.

---

## 5. Q&A bank

**Q: What is mode collapse and did you see it?**
A: Mode collapse is when the Generator learns to produce only a few types of
outputs (or even a single output) that reliably fool the Discriminator, instead
of covering the full diversity of the real data distribution. You can watch for
it by generating a large batch and checking if many samples look near-identical.
Mitigations include: adding minibatch discrimination, using Wasserstein loss
(WGAN-GP), or simply using a smaller learning rate / more careful architecture,
which is what this project relies on.

**Q: How do you know training is going well, if there's no "accuracy" metric like
a normal classifier?**
A: GAN losses alone are notoriously unreliable indicators — a "good-looking"
loss curve doesn't guarantee good images, and vice versa. In this project I track
Discriminator accuracy on real vs. fake batches (in the "Training Curves" tab):
ideally both hover around 50-70%, indicating the two networks are roughly
balanced. If Discriminator accuracy shoots to ~100%, it's overpowering the
Generator, which usually stalls learning. I also visually inspect sample grids
saved every epoch. For a rigorous quantitative measure, the standard tool is the
**FID (Fréchet Inception Distance)** score, which I mention as a future
extension.

**Q: Why not just use a VAE (Variational Autoencoder) instead?**
A: VAEs optimize a reconstruction loss plus a KL-divergence regularizer, which
tends to produce blurrier images because the loss rewards "averaging" over
plausible reconstructions. GANs use an adversarial loss instead of a
pixel-wise reconstruction loss, so they tend to produce sharper, more realistic
images — at the cost of being harder to train stably (no single loss number
that reliably says "better").

**Q: What's the difference between this and a Conditional GAN?**
A: This is an *unconditional* GAN — you generate a random clothing item, but
can't choose which of the 10 categories. A Conditional GAN (cGAN) would
concatenate a one-hot class label (or an embedding of it) to the noise vector
fed to G, and also feed the label to D, so both networks are conditioned on the
class. That would let a user pick "generate a sneaker" specifically.

**Q: How did you validate you didn't just memorize the training set?**
A: The latent interpolation tab is exactly for this — if the Generator had
simply memorized training images, morphing between two random latent vectors
would produce abrupt jumps between memorized images rather than smooth,
gradual transitions. Smooth interpolation is evidence the model learned a
continuous, meaningful latent space rather than memorizing.

**Q: Why Adam and not SGD?**
A: Adam's adaptive per-parameter learning rates and momentum handle the
non-stationary, adversarial loss landscape of GAN training much better than
plain SGD, which tends to be far less stable here. The DCGAN paper's specific
tuning (lr=0.0002, β1=0.5) is chosen to reduce oscillation between G and D.

**Q: What would you change with more time/compute?**
A: (1) Move to WGAN-GP for more stable training and a more interpretable loss
signal; (2) add a Conditional GAN so users can pick a class; (3) train on a
harder dataset like CelebA to show the architecture scales; (4) add an FID
score for objective evaluation; (5) deploy the Streamlit app publicly so it's
a live demo link on my resume, not just code.

**Q: Walk me through what happens in one training step.**
A: (1) Sample a real batch of images and a batch of random noise vectors.
(2) Generate fake images from the noise. (3) Update D: compute its loss on
real images (labeled real) plus its loss on the *detached* fake images
(labeled fake), backprop, step the D optimizer. (4) Update G: run the fake
images (not detached this time) back through the now-updated D, compute loss
using *real* labels (the non-saturating trick), backprop, step the G
optimizer. Repeat for every batch, every epoch.

**Q: Why detach the fake images when training the Discriminator?**
A: `.detach()` stops gradients from flowing back into the Generator during the
Discriminator's update step. At that point we only want to update D's weights,
not G's — G's turn comes right after, using a fresh forward pass through D.

---

## 6. If asked to extend the project live in the interview

Good, scoped, explainable extensions if asked to code something on the spot:
- Add a `--conditional` flag: concatenate a one-hot label to the latent vector
  in `Generator.forward` and to the image channels in `Discriminator.forward`.
- Add early stopping / best-checkpoint saving based on a held-out FID proxy.
- Swap `BCELoss` for the Wasserstein loss + gradient penalty (WGAN-GP) — a
  favorite "explain the tradeoffs" interview question.
