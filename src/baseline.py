"""Majority class baseline: predict the most common train class for everything."""

import json
from pathlib import Path

import numpy as np

from dataset import load_eurosat
from split import SPLIT_PATH, load_split

RESULTS_PATH = Path("results/baseline.json")


def majority_class_accuracy(train_labels, eval_labels):
    majority = int(np.bincount(train_labels).argmax())
    acc = float((eval_labels == majority).mean())
    return majority, acc


def main():
    dataset = load_eurosat()
    labels = np.array(dataset.targets)
    split = load_split(SPLIT_PATH)

    majority, test_acc = majority_class_accuracy(labels[split["train"]], labels[split["test"]])
    _, val_acc = majority_class_accuracy(labels[split["train"]], labels[split["val"]])
    name = dataset.classes[majority]

    print(f"Majority class in train: {name}")
    print(f"Baseline accuracy, val:  {val_acc:.4f}")
    print(f"Baseline accuracy, test: {test_acc:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {"majority_class": name, "val_accuracy": val_acc, "test_accuracy": test_acc},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
