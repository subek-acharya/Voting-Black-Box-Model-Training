# ResNet & VGG Training

Training code for the standard (non-spiking) ResNet and VGG CNN classifiers. These
architectures are trained on CIFAR-10 with support for white-box adversarial evaluation
(FGSM, PGD).

Full source: [pytorch-train-eval-adversarial-robustness-cifar10](https://github.com/subek-acharya/pytorch-train-eval-adversarial-robustness-cifar10).

## Models

| Model | Builder | Dataset |
| --- | --- | --- |
| ResNet-18 | `resnet.PreActResNet18()` | CIFAR-10 |
| VGG-16 | `vgg.VGG('VGG16')` | CIFAR-10 |

The reference repo also provides DenseNet-121 and GoogLeNet under the same training setup.

## Getting the Code

```bash
git clone https://github.com/subek-acharya/pytorch-train-eval-adversarial-robustness-cifar10.git
cd pytorch-train-eval-adversarial-robustness-cifar10
pip install -r requirement.txt
```

## Training

Select the model in `train.py` (uncomment the desired builder), then run:

```bash
# ResNet-18
model = resnet.PreActResNet18().to(device)

# VGG-16
# model = vgg.VGG('VGG16').to(device)

python train.py
```

### Default Training Parameters

| Parameter | Value |
| --- | --- |
| Epochs | 200 |
| Batch Size | 64 |
| Learning Rate | 0.01 (Cosine Annealing) |
| Optimizer | SGD, momentum 0.9 |
| Weight Decay | 5e-4 |

## Evaluation

Set the checkpoint path and model in `main.py`, then run:

```bash
modelDir = "./checkpoint/resnet18.pth"
model = resnet.PreActResNet18().to(device)

results = evaluation.evaluate_model(device, model, valLoader)
evaluation.print_evaluation_results(results)

python main.py
```

## Adversarial Attacks

White-box FGSM and PGD attacks are run from `main.py` via `AttackWrappersWhiteBox`.

### Attack Parameters

| Parameter | Value |
| --- | --- |
| epsilonMax | 0.031 |
| clipMin / clipMax | 0.0 / 1.0 |
| numSteps (PGD) | 20 |
| epsilonStep | epsilonMax / numSteps |

## Results (Clean CIFAR-10)

| Model | Test Accuracy | Parameters |
| --- | --- | --- |
| ResNet-18 | 95.17% | 11.17M |
| VGG-16 | 93.33% | 14.73M |
