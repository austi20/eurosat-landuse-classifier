"""The installed torch and torchvision match the pins in requirements.txt."""

import torch
import torchvision


def test_torch_pin():
    assert torch.__version__.split("+")[0] == "2.14.0"


def test_torchvision_pin():
    assert torchvision.__version__.split("+")[0] == "0.29.0"
