"""Small CNN trained from scratch, and the pretrained ResNet18 it is compared against."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 64 -> 32 -> 16 -> 8 spatial
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(num_classes=10, pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    # ImageNet's 1000-way head swapped for a fresh 10-way one
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
