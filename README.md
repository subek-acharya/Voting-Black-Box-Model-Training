# Model Training for Black-Box Adversarial Attacks on Election Systems

This repository is the training hub for the image classifiers used in our black-box
adversarial attack framework on machine-learning based election systems. It gathers the
training code, configuration, and instructions needed to reproduce each model from
scratch. The trained models are then attacked in the companion
[Voting-Black-Box](https://github.com/subek-acharya/Voting-Black-Box) repository.

# Software Installation

We use the following software packages:

<ul>
  <li>python==3.10</li>
  <li>torch>=2.0</li>
  <li>torchvision>=0.15</li>
  <li>numpy</li>
  <li>scipy</li>
  <li>spikingjelly (Spiking Neural Networks)</li>
  <li>protopnext / cosine-is-almost (Explainable AI / ProtoPNet)</li>
</ul>

```bash
git clone https://github.com/subek-acharya/Voting-Black-Box-Model-Training.git
cd Voting-Black-Box-Model-Training
pip install -r requirements.txt
```

# Models

We train the following architectures for ballot bubble classification. Each model has a
dedicated training folder under `models/`.

| Model | Family | Training Folder | Source Repo |
| --- | --- | --- | --- |
| ResNet-18 | CNN | `models/resnet-vgg/` | [pytorch-train-eval-adversarial-robustness-cifar10](https://github.com/subek-acharya/pytorch-train-eval-adversarial-robustness-cifar10) |
| VGG-16 | CNN | `models/resnet-vgg/` | [pytorch-train-eval-adversarial-robustness-cifar10](https://github.com/subek-acharya/pytorch-train-eval-adversarial-robustness-cifar10) |
| Spiking VGG16 | Spiking Neural Network | `models/spiking-neural-network/` | [Spiking-Neural-Network](https://github.com/subek-acharya/Spiking-Neural-Network) |
| Spiking ResNet20 | Spiking Neural Network | `models/spiking-neural-network/` | [Spiking-Neural-Network](https://github.com/subek-acharya/Spiking-Neural-Network) |
| ProtoPNet-VGG16 | Explainable AI | `models/explainable-ai/` | [Explainable-AI](https://github.com/subek-acharya/Explainable-AI) |
| ProtoPNet-ResNet20 | Explainable AI | `models/explainable-ai/` | [Explainable-AI](https://github.com/subek-acharya/Explainable-AI) |
| CaiT | Vision Transformer | `models/cait-svm/` | in-repo (`model_architecture/`) |
| Multi-output SVM | SVM | `models/cait-svm/` | in-repo (`model_architecture/`) |

These join the broader set of eight classifiers attacked in the main framework: ResNet20,
VGG16, CaiT, Multi-output SVM, Spiking ResNet20, Spiking VGG16, ProtoPNet-ResNet20, and
ProtoPNet-VGG16.

# Repository Layout

```
Voting-Black-Box-Model-Training/
├── README.md
├── requirements.txt
├── data/                           # datasets (git-ignored, supplied locally)
├── checkpoints/                    # trained model weights (git-ignored)
├── docs/
│   ├── SETUP.md                    # environment and dataset setup
│   └── TRAINING.md                 # training workflow and conventions
└── models/
    ├── resnet-vgg/                 # ResNet-18 / VGG-16 CNN training (CIFAR-10)
    │   └── README.md
    ├── spiking-neural-network/     # Spiking VGG16 / ResNet20 training (SpikingJelly)
    │   └── README.md
    ├── explainable-ai/             # ProtoPNet VGG16 / ResNet20 training
    │   └── README.md
    └── cait-svm/                   # CaiT + Multi-output SVM architectures (in-repo code)
        ├── README.md
        └── model_architecture/
            ├── cait.py
            └── svm.py
```

# Dataset

Models are trained on the Combined and OnlyBubbles grayscale voting datasets
(single-channel 40×50 ballot-bubble images, binary classification). Datasets live under
`data/`, which is git-ignored and supplied locally. The datasets are available at
<https://zenodo.org/records/18344123>. The Spiking VGG16 CIFAR-10 reference model instead
uses the standard CIFAR-10 dataset, which is downloaded automatically.

# Training

Each model family has its own instructions. Start here:

<ul>
  <li><a href="models/resnet-vgg/README.md">ResNet &amp; VGG training</a></li>
  <li><a href="models/spiking-neural-network/README.md">Spiking Neural Network training</a></li>
  <li><a href="models/explainable-ai/README.md">Explainable AI (ProtoPNet) training</a></li>
  <li><a href="models/cait-svm/README.md">CaiT &amp; Multi-output SVM</a></li>
  <li><a href="docs/SETUP.md">Environment and dataset setup</a></li>
  <li><a href="docs/TRAINING.md">Training workflow and conventions</a></li>
</ul>

# Contact

For questions or concerns please contact the author at: subekacharya@uri.edu
