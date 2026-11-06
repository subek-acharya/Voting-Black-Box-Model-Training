# Training Workflow and Conventions

This document describes the shared workflow for training the classifiers in this
repository and the conventions each model folder follows.

## Workflow

1. **Set up the environment and datasets** — see [SETUP.md](SETUP.md).
2. **Pick a model family** under `models/` and read its `README.md`.
3. **Train** using the model's training script.
4. **Evaluate** on the clean validation / test set.
5. **Save the checkpoint** to `checkpoints/` (git-ignored) and record results.

## Conventions

### Directory Layout per Model

Each model family lives under `models/<family>/` with its own `README.md` describing the
training and evaluation entry points. The heavy, self-contained implementations stay in
their dedicated source repositories (linked from each README); this repo is the training
hub that documents and coordinates them.

| Family | Folder | Source Repo |
| --- | --- | --- |
| Spiking Neural Network | `models/spiking-neural-network/` | [Spiking-Neural-Network](https://github.com/subek-acharya/Spiking-Neural-Network) |
| Explainable AI (ProtoPNet) | `models/explainable-ai/` | [Explainable-AI](https://github.com/subek-acharya/Explainable-AI) |

### Datasets

- Voter models train on the OnlyBubbles / Combined grayscale datasets (1×40×50, binary).
- The Spiking VGG16 CIFAR-10 model is a reference/sanity model on CIFAR-10 (32×32, 10 class).
- Datasets live under `data/` (git-ignored); see [SETUP.md](SETUP.md).

### Checkpoints

- Save trained weights as `.pth` under `checkpoints/`.
- Use descriptive names, e.g. `spiking_vgg16_bn_voter.pth`, `Explainable_ResNet20.pth`.
- Checkpoints are git-ignored — do not commit large binaries.

### Reproducibility

- Record the exact training parameters used (see each model README for defaults).
- For Spiking Neural Networks, the timestep count `T` must match between training and
  evaluation.
- Fix random seeds when comparing runs.

## Relationship to the Attack Framework

Models trained here are consumed by the black-box attack framework in
[Voting-Black-Box](https://github.com/subek-acharya/Voting-Black-Box), which runs
multi-norm adversarial attacks against the eight voting-system classifiers. Keep model
names and checkpoint conventions consistent so trained weights drop into the attack
framework's `ModelFactory` without renaming.
