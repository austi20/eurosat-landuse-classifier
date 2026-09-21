"""Smoke tests: a fresh checkout has the pinned stack and can reach EuroSAT.

These do not download the dataset. They assert the pins in requirements.txt are
what actually got installed, so a checkout that silently resolved a different
torch does not get discovered halfway through a training run.
"""

import torch
import torchvision
from torchvision.datasets import EuroSAT


def test_torch_pin():
    assert torch.__version__.split("+")[0] == "2.14.0"


def test_torchvision_pin():
    assert torchvision.__version__.split("+")[0] == "0.29.0"


def test_eurosat_dataset_available():
    assert issubclass(EuroSAT, torchvision.datasets.ImageFolder)


def test_resnet18_imagenet_weights_enumerated():
    """The transfer-learning arm needs pretrained weights to exist by name."""
    from torchvision.models import ResNet18_Weights

    assert ResNet18_Weights.IMAGENET1K_V1 is not None
