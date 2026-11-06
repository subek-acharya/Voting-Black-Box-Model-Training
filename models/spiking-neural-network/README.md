# Spiking Neural Network Training

Training code for Spiking VGG16 and Spiking ResNet20 classifiers on the voting (ballot
bubble) datasets, built on the [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
framework. Models are trained from scratch with surrogate gradient learning.

Full source: [Spiking-Neural-Network](https://github.com/subek-acharya/Spiking-Neural-Network).

## Models

| Model | Input | Classes | Dataset |
| --- | --- | --- | --- |
| Spiking VGG16-BN | 32×32×3 | 10 | CIFAR-10 (reference) |
| Spiking VGG16 (Voter) | 1×40×50 | 2 | Combined / OnlyBubbles |
| Spiking ResNet20 (Voter) | 1×40×50 | 2 | Combined / OnlyBubbles |

## Getting the Code

```bash
git clone https://github.com/subek-acharya/Spiking-Neural-Network.git
cd Spiking-Neural-Network
pip install -r requirements.txt
```

## Training

```bash
# Spiking VGG16 on CIFAR-10 (reference model)
python train_snn_vgg_cifar.py

# Spiking VGG16 on the Voter dataset (1×40×50 grayscale ballots)
python train_snn_vgg_voter.py

# Spiking ResNet20 on the Voter dataset
python train_snn_resnet_voter.py
```

### Default Training Parameters

| Parameter | Value |
| --- | --- |
| Timesteps (T) | 4 |
| Epochs | 200 |
| Batch Size | 64 |
| Learning Rate | 0.01 (Cosine Annealing) |
| Optimizer | SGD, momentum 0.9 |
| Weight Decay | 5e-4 |
| Spiking Neuron | IFNode |
| Surrogate Function | ATan |

## Evaluation

```bash
python evaluate_cifar.py     # CIFAR-10 test set
python evaluate_voter.py     # all Voter SNN models
```

## Notes

- **Timesteps (T)** must match between training and evaluation; the model learns temporal
  dynamics specific to `T=4`.
- Always call `functional.reset_net(model)` after each batch to reset membrane potentials.
- Use `functional.set_step_mode(model, 'm')` for multi-step `[T, N, C, H, W]` input.

## Reference

- [SpikingJelly Documentation](https://spikingjelly.readthedocs.io/)
- Trained checkpoints belong under the repo-level `checkpoints/` folder (git-ignored).
