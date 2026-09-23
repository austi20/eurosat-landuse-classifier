import torch

from model import SmallCNN


def test_small_cnn_output_shape():
    model = SmallCNN(num_classes=10)
    x = torch.zeros(4, 3, 64, 64)
    assert model(x).shape == (4, 10)
