"""Score the small CNN on the test split. Does not change which model was chosen."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from dataset import load_eurosat, load_images
from evaluate import predict_test
from split import SPLIT_PATH, load_split

RESULTS_PATH = Path("results/small_cnn_test.json")


def main():
    dataset = load_eurosat()
    images = torch.from_numpy(load_images(dataset))
    targets = np.array(dataset.targets)
    split = load_split(SPLIT_PATH, dataset.targets)
    num_classes = len(dataset.classes)

    # Val check confirms the checkpoint is the run in the README
    val_preds = predict_test("small_cnn", images, {**split, "test": split["val"]}, num_classes).numpy()
    val_acc = float((val_preds == targets[split["val"]]).mean())
    print(f"small_cnn val accuracy {val_acc:.4f}")

    test_y = targets[split["test"]]
    test_preds = predict_test("small_cnn", images, split, num_classes).numpy()
    n = len(test_y)
    acc = float((test_preds == test_y).mean())
    half_width = 1.96 * math.sqrt(acc * (1 - acc) / n)
    print(f"small_cnn test accuracy {acc:.4f}  (95% CI {acc - half_width:.4f} to {acc + half_width:.4f}, n={n})")

    per_class = {}
    for c, name in enumerate(dataset.classes):
        mask = test_y == c
        per_class[name] = float((test_preds[mask] == c).mean())

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "val_accuracy": val_acc,
            "test_accuracy": acc,
            "test_accuracy_95ci": [acc - half_width, acc + half_width],
            "n_test": n,
            "test_accuracy_by_class": per_class,
        }, f, indent=2)


if __name__ == "__main__":
    main()
