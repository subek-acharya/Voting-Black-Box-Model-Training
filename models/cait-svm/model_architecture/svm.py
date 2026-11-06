import torch
import torch.nn as nn

class SVMPyTorch(nn.Module):
    def __init__(self, input_dim, C=1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # Binary: scores for +1 class
        self.C = C  # Regularization tradeoff

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten [B,C,H,W] to [B, C*H*W]
        return self.linear(x).squeeze(1)  # [B] scores

    def hinge_loss(self, outputs, labels):
        # labels: -1/+1
        return torch.mean(torch.clamp(1 - labels * outputs, min=0))

    def regularization(self):
        return 0.5 * torch.norm(self.linear.weight)**2  # 1/2 ||w||^2 for max margin

    def loss(self, outputs, labels):
        return self.C * self.hinge_loss(outputs, labels) + self.regularization()