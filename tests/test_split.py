"""Split and baseline tests on synthetic labels, so no download is needed."""

import numpy as np
import pytest

from baseline import majority_class_accuracy
from split import load_split, save_split, stratified_split


def make_labels():
    # Unbalanced on purpose: 3000, 2000, 1000
    return np.array([0] * 3000 + [1] * 2000 + [2] * 1000)


def test_split_sizes_are_70_15_15():
    labels = make_labels()
    train, val, test = stratified_split(labels, seed=42)
    assert len(train) == 4200
    assert len(val) == 900
    assert len(test) == 900


def test_split_is_disjoint_and_complete():
    labels = make_labels()
    train, val, test = stratified_split(labels, seed=42)
    all_idx = np.concatenate([train, val, test])
    assert len(set(all_idx)) == len(labels)
    assert sorted(all_idx) == list(range(len(labels)))


def test_split_keeps_class_proportions():
    labels = make_labels()
    for part in stratified_split(labels, seed=42):
        counts = np.bincount(labels[part], minlength=3)
        shares = counts / counts.sum()
        assert np.allclose(shares, [0.5, 1 / 3, 1 / 6], atol=0.01)


def test_same_seed_same_split():
    labels = make_labels()
    first = stratified_split(labels, seed=7)
    second = stratified_split(labels, seed=7)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_different_seed_different_split():
    labels = make_labels()
    train_a, _, _ = stratified_split(labels, seed=1)
    train_b, _, _ = stratified_split(labels, seed=2)
    assert not np.array_equal(np.sort(train_a), np.sort(train_b))


def test_split_file_round_trip(tmp_path):
    labels = np.array([0, 0, 1, 1, 2, 2])
    paths = [f"C{label}/img_{i}.jpg" for i, label in enumerate(labels)]
    train, val, test = np.array([0, 2, 4]), np.array([1, 3]), np.array([5])
    out = tmp_path / "split.csv"

    save_split(out, paths, labels, train, val, test)
    loaded = load_split(out, labels)

    assert np.array_equal(loaded["train"], train)
    assert np.array_equal(loaded["val"], val)
    assert np.array_equal(loaded["test"], test)


def test_load_split_rejects_reordered_dataset(tmp_path):
    labels = np.array([0, 0, 1, 1])
    paths = [f"img_{i}.jpg" for i in range(4)]
    out = tmp_path / "split.csv"
    save_split(out, paths, labels, np.array([0, 1]), np.array([2]), np.array([3]))

    reordered = np.array([1, 1, 0, 0])
    with pytest.raises(ValueError):
        load_split(out, reordered)


def test_majority_baseline_uses_train_majority():
    train_labels = np.array([0, 0, 0, 1])
    # Test majority is 1, but the baseline only knows train
    test_labels = np.array([0, 1, 1, 1])
    majority, acc = majority_class_accuracy(train_labels, test_labels)
    assert majority == 0
    assert acc == 0.25
