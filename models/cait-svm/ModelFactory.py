import torch
import torch.nn as nn
from typing import Optional

from model_architecture.cait import CaiT
from model_architecture.svm import SVMPyTorch


# ----------------- MODEL FACTORY -----------------
# Mirrors the style of the attack repo's load-only ModelFactory, but this one
# BUILDS fresh models (no checkpoint loading) for training from scratch.
class ModelFactory:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def build_cait(self, num_classes: int = 2) -> nn.Module:
        """Construct a CaiT for the 1x40x50 grayscale voter inputs."""
        model = CaiT(
            image_size=(40, 50),
            patch_size=5,
            num_classes=num_classes,
            num_channels=1,
            dim=512,
            depth=16,
            cls_depth=2,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05,
        ).to(self.device)
        return model

    def build_svm(self, input_dim: int = 2000, C: float = 1.0) -> nn.Module:
        """Construct a linear SVM over flattened 1x40x50 = 2000-dim inputs."""
        model = SVMPyTorch(input_dim, C).to(self.device)
        return model

    def get_model(self, model_name: str, **kwargs) -> nn.Module:
        name = model_name.lower()
        if "cait" in name:
            return self.build_cait(**kwargs)
        elif "svm" in name:
            return self.build_svm(**kwargs)
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")
