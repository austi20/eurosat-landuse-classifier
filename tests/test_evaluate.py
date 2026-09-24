import numpy as np

from evaluate import choose_model, top_confusions


def test_choose_model_picks_higher_validation():
    assert choose_model({"small_cnn": 0.89, "resnet18": 0.97}) == "resnet18"
    assert choose_model({"small_cnn": 0.91, "resnet18": 0.90}) == "small_cnn"


def test_top_confusions_skips_diagonal_and_zeros():
    cm = np.array([[50, 3, 0],
                   [1, 40, 7],
                   [0, 0, 60]])
    names = ["A", "B", "C"]
    assert top_confusions(cm, names, n=5) == [("B", "C", 7), ("A", "B", 3), ("B", "A", 1)]
