# Explainable AI (ProtoPNet) Training

Training code for interpretable ballot-bubble classifiers using the ProtoPNet architecture
with ResNet-20 and VGG-16 backbones. The models learn class-specific prototypes that
correspond to real image patches, giving interpretable "this looks like that" predictions.

Full source: [Explainable-AI](https://github.com/subek-acharya/Explainable-AI).

## Models

| Model | Backbone | Prototypes | Total Params |
| --- | --- | --- | --- |
| ProtoPNet-ResNet20 | ResNet-20 | 40 (20/class) | 709,680 |
| ProtoPNet-VGG16 | VGG-16 | 40 (20/class) | 15,267,856 |

Input: single-channel 40×50 grayscale ballot images, binary classification. Prototype layer
uses cosine similarity between projected features and learned prototypes.

## Getting the Code

```bash
git clone https://github.com/subek-acharya/Explainable-AI.git
cd Explainable-AI

# ProtoPNet framework is supplied as a submodule
git clone https://github.com/protopnext/cosine-is-almost.git

pip install torch torchvision numpy
```

The expected layout (ProtoPNet framework under `cosine-is-almost/protopnext/protopnet/`):

```
Explainable-AI/
├── ModelFactory.py              # loads ProtoPNet (expv2 / explainable) checkpoints
├── evaluate_xAI.py              # evaluate on all datasets
├── check_prototype_params.py    # inspect prototype layer parameters
├── utils.py
├── checkpoint/
│   ├── Explainable_ResNet20.pth
│   └── Explainable_VGG16.pth
├── models/
│   └── architecture/ResNet.py   # custom ResNet-20 backbone
└── cosine-is-almost/
    └── protopnext/protopnet/
```

## Evaluation

```bash
python evaluate_xAI.py            # evaluate ProtoPNet models on all datasets
python check_prototype_params.py  # inspect prototype layer parameters
```

## Results

| Model | OnlyBubbles Val | Combined Val |
| --- | --- | --- |
| xAI-ResNet20 | 1.0000 | 0.9959 |
| xAI-VGG16 | 1.0000 | 0.9954 |

## Reference

1. Chen, Chaofan, et al. [*This looks like that: deep learning for interpretable image
   recognition.*](https://arxiv.org/abs/1806.10574) NeurIPS 32 (2019).
