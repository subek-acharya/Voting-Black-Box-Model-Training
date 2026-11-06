# CaiT & Multi-output SVM

Model architectures for two of the voting-system classifiers: the **CaiT** vision
transformer and a **Multi-output SVM**. Unlike the other model families, these do not have
a standalone source repository — the architecture code lives here directly under
`model_architecture/`.

## Models

| Model | File | Description |
| --- | --- | --- |
| CaiT | `model_architecture/cait.py` | Class-Attention in Image Transformers, adapted for grayscale ballot images |
| Multi-output SVM | `model_architecture/svm.py` | Linear SVM with hinge loss and max-margin regularization |

## CaiT

A Class-Attention in Image Transformers (CaiT) implementation, adapted from
[lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py).
The adaptation supports a configurable channel count (`num_channels`) and a non-square
`image_size`, so it accepts the single-channel 40×50 grayscale ballot inputs used by the
voting datasets.

Key constructor arguments (`CaiT`):

| Argument | Purpose |
| --- | --- |
| `image_size` | `(H, W)` tuple, e.g. `(40, 50)` |
| `patch_size` | patch side length used for tokenization |
| `num_channels` | input channels (1 for grayscale ballots) |
| `num_classes` | output classes (2 for ballot bubble classification) |
| `dim`, `depth`, `cls_depth`, `heads`, `mlp_dim` | transformer capacity |

The model uses separate patch and class-attention transformers, LayerScale, and talking-
heads attention (pre- and post-softmax head mixing).

## Multi-output SVM

`SVMPyTorch` is a linear SVM expressed as a `torch.nn.Module` for binary (+1 / −1)
classification:

- `forward(x)` flattens `[B, C, H, W]` inputs and returns per-sample scores.
- `hinge_loss(outputs, labels)` computes the margin hinge loss (labels in {−1, +1}).
- `regularization()` adds the `½‖w‖²` max-margin term.
- `loss(outputs, labels)` combines them as `C · hinge + regularization`, where `C` is the
  regularization tradeoff set at construction.

## Usage

```python
from models.cait_svm.model_architecture.cait import CaiT
from models.cait_svm.model_architecture.svm import SVMPyTorch

# CaiT for grayscale ballots (1×40×50, binary)
cait = CaiT(
    image_size=(40, 50), patch_size=5, num_channels=1, num_classes=2,
    dim=128, depth=6, cls_depth=2, heads=4, mlp_dim=256,
)

# Multi-output SVM on flattened 1×40×50 input
svm = SVMPyTorch(input_dim=1 * 40 * 50, C=1.0)
```

> Note: these files provide the model definitions. Training/evaluation for CaiT and the
> Multi-output SVM is driven from the attack framework
> ([Voting-Black-Box](https://github.com/subek-acharya/Voting-Black-Box)), where both
> architectures are registered in its `ModelFactory`.

## Training

A small, self-contained training subsystem lives alongside the architectures. Run it from
this folder so the flat imports and default relative paths resolve:

```bash
cd models/cait-svm
python train.py --model cait   # train the CaiT classifier
python train.py --model svm    # train the linear SVM
```

`train.py` builds the model through the local `ModelFactory`, loads data with the loaders
in `utils.py`, runs the training loop, and saves the best-validation checkpoint under
`checkpoints/` (`checkpoints/cait_voter.pth` or `checkpoints/svm_voter.pth`), holding
`{"state_dict", "acc", "epoch"}`.

### Default hyperparameters

| Flag | Default | Notes |
| --- | --- | --- |
| `--epochs` | `200` | number of training epochs |
| `--batch-size` | `64` | mini-batch size |
| `--lr` | `0.01` | SGD learning rate |
| `--data-dir` | `./data` | where the Voter `.pth` datasets live |
| `--checkpoint-dir` | `./checkpoints` | where checkpoints are written |
| `--device` | auto | `cuda` if available, else `cpu` |

Optimizer and schedule are fixed: SGD with `momentum=0.9`, `weight_decay=5e-4`, and
`CosineAnnealingLR(T_max=epochs)`. The two models differ only in their loss and accuracy:

- **CaiT** — `CrossEntropyLoss` on the `[B, 2]` logits, argmax accuracy against `{0,1}`
  labels.
- **SVM** — `model.loss()` (hinge + ½‖w‖² margin) with labels mapped `{0,1} -> {-1,+1}`,
  and sign-match accuracy (`sign(score) == label`).

### Data

Training reads two torch `.pth` dicts with keys `data` (`[N, 1, 40, 50]` float) and
`binary_labels` (long `{0,1}`):

- `kaleel_final_dataset_train_Combined_Grayscale.pth` (train)
- `kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth` (validation)

These Voter datasets are not downloaded by the scripts. Place them under the data dir
(default `./data`, available from Zenodo) before training can run.

> Actual training is intended to run on the Unity cluster, where `torch`, `einops`, a CUDA
> GPU, and the Zenodo datasets are available. The code in this folder is verified locally by
> static checks (parsing and manual signature/shape review) only.
