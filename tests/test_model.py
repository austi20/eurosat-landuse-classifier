import torch

from model import SmallCNN, build_resnet18
from train_resnet import prepare


def test_small_cnn_output_shape():
    model = SmallCNN(num_classes=10)
    x = torch.zeros(4, 3, 64, 64)
    assert model(x).shape == (4, 10)


def test_resnet18_has_ten_way_head():
    # pretrained=False so the test does not download weights
    model = build_resnet18(num_classes=10, pretrained=False)
    model.eval()
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 10)


def test_prepare_resizes_and_normalizes():
    batch = torch.full((2, 3, 64, 64), 255, dtype=torch.uint8)
    x = prepare(batch)
    assert x.shape == (2, 3, 224, 224)
    # A white pixel is (1 - mean) / std in every channel
    expected = (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(x[0, :, 100, 100], expected)
