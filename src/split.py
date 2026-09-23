"""Seeded, stratified 70/15/15 split of EuroSAT, saved to a CSV."""

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from dataset import class_counts, load_eurosat

SEED = 42
SPLIT_PATH = Path("splits/eurosat_split_seed42.csv")


def stratified_split(labels, seed):
    indices = np.arange(len(labels))
    train, rest = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=seed
    )
    # Halve the 30% remainder into val and test
    val, test = train_test_split(
        rest, test_size=0.50, stratify=labels[rest], random_state=seed
    )
    return train, val, test


def save_split(path, paths, labels, train, val, test):
    split_of = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        for i in part:
            split_of[int(i)] = name

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "path", "label", "split"])
        for i in range(len(labels)):
            writer.writerow([i, paths[i], int(labels[i]), split_of[i]])


def load_split(path):
    parts = {"train": [], "val": [], "test": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parts[row["split"]].append(int(row["index"]))
    return {name: np.array(idx) for name, idx in parts.items()}


def main():
    dataset = load_eurosat()
    labels = np.array(dataset.targets)
    # Relative paths so the file does not depend on where data/ lives
    root = Path(dataset.root)
    paths = [Path(p).relative_to(root).as_posix() for p, _ in dataset.samples]

    print("Per-class counts, full dataset:")
    counts = class_counts(labels, dataset.classes)
    for name, n in counts.items():
        print(f"  {name:22s} {n:5d}  {n / len(labels):6.2%}")
    print(f"  {'total':22s} {len(labels):5d}")

    train, val, test = stratified_split(labels, seed=SEED)
    save_split(SPLIT_PATH, paths, labels, train, val, test)
    print(f"\nSaved {SPLIT_PATH}: train {len(train)}, val {len(val)}, test {len(test)}")

    print("\nPer-class counts by split (train / val / test):")
    for c, name in enumerate(dataset.classes):
        n_train = int((labels[train] == c).sum())
        n_val = int((labels[val] == c).sum())
        n_test = int((labels[test] == c).sum())
        print(f"  {name:22s} {n_train:5d} {n_val:5d} {n_test:5d}")


if __name__ == "__main__":
    main()
