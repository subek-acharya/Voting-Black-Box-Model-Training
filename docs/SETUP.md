# Environment and Dataset Setup

This guide covers preparing the environment and datasets needed to train the voting-system
classifiers in this repository.

## 1. Python Environment

We target Python 3.10. Using a virtual environment is recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU

Training is designed for CUDA GPUs. Install the `torch` / `torchvision` build matching
your CUDA version from <https://pytorch.org/get-started/locally/>. CPU-only runs work but
are slow, especially for the multi-timestep Spiking Neural Networks.

## 2. Datasets

| Dataset | Used By | Shape | Source |
| --- | --- | --- | --- |
| OnlyBubbles | SNN Voter, ProtoPNet | 1×40×50 grayscale | [Zenodo 18344123](https://zenodo.org/records/18344123) |
| Combined | SNN Voter, ProtoPNet | 1×40×50 grayscale | [Zenodo 18344123](https://zenodo.org/records/18344123) |
| CIFAR-10 | Spiking VGG16 (reference) | 32×32×3 | auto-downloaded |

Download the voting datasets and place them under `data/`:

```
data/
├── OnlyBubbles/
└── Combined/
```

`data/` is git-ignored. CIFAR-10 is downloaded automatically into the SNN repo's `data/`
folder on first training run.

## 3. Checkpoints

Trained weights (`.pth`) are written to `checkpoints/`, which is git-ignored. Large model
binaries are not committed to this repository; share them via the dataset/model release
channel instead.

## 4. Model-Specific Setup

Some models require extra setup beyond `requirements.txt`:

<ul>
  <li><b>Explainable AI (ProtoPNet)</b> needs the <code>cosine-is-almost</code> /
      <code>protopnext</code> framework. See
      <a href="../models/explainable-ai/README.md">models/explainable-ai/README.md</a>.</li>
  <li><b>Spiking Neural Networks</b> require <code>spikingjelly</code> (already in
      <code>requirements.txt</code>). See
      <a href="../models/spiking-neural-network/README.md">models/spiking-neural-network/README.md</a>.</li>
</ul>
