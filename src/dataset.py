"""EuroSAT loading, with the decoded images cached as one uint8 array."""

from pathlib import Path

import numpy as np
from torchvision.datasets import EuroSAT

DATA_ROOT = Path("data")
CACHE_PATH = DATA_ROOT / "eurosat_rgb_uint8.npy"


def load_eurosat():
    return EuroSAT(root=str(DATA_ROOT), download=True)


def class_counts(labels, class_names):
    counts = np.bincount(labels, minlength=len(class_names))
    return {name: int(n) for name, n in zip(class_names, counts)}


def load_images(dataset):
    """All images as (N, 3, 64, 64) uint8, in dataset index order."""
    if CACHE_PATH.exists():
        images = np.load(CACHE_PATH)
        # Partial download would desync images and labels
        if len(images) != len(dataset):
            raise ValueError(f"{CACHE_PATH} has {len(images)} images, dataset has {len(dataset)}. Delete it and rerun.")
        return images

    # Decoding 27k JPEGs every epoch is the slow part on CPU
    images = np.zeros((len(dataset), 3, 64, 64), dtype=np.uint8)
    for i in range(len(dataset)):
        img, _ = dataset[i]
        images[i] = np.asarray(img.convert("RGB")).transpose(2, 0, 1)
    np.save(CACHE_PATH, images)
    return images
