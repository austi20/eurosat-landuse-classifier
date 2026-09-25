"""Pick the better model on validation, then score it on the test split. Run this once."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from sklearn.metrics import confusion_matrix

import train_resnet
import train_scratch
from dataset import load_eurosat, load_images
from model import SmallCNN, build_resnet18
from split import SPLIT_PATH, load_split

RESULTS_PATH = Path("results/test.json")
FIGURE_PATH = Path("figures/confusion_matrix.png")

# Class pairs named in the plan as the likely confusions, checked either way round
PAIRS_OF_INTEREST = [("Highway", "River"), ("PermanentCrop", "HerbaceousVegetation")]


def choose_model(val_accuracies):
    """Name of the model with the highest validation accuracy."""
    return max(val_accuracies, key=val_accuracies.get)


def top_confusions(cm, class_names, n=5):
    """The n largest off-diagonal cells as (true, predicted, count)."""
    cells = [(cm[i, j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j]
    cells.sort(reverse=True)
    return [(class_names[i], class_names[j], int(count)) for count, i, j in cells[:n] if count > 0]


def predict_test(name, images, split, num_classes):
    if name == "resnet18":
        model = build_resnet18(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(train_resnet.CHECKPOINT_PATH))
        model = model.to(memory_format=torch.channels_last)
        return train_resnet.predict(model, images[split["test"]])

    # Same train-only channel stats the scratch CNN was trained with
    mean, std = train_scratch.channel_stats(images[split["train"]])
    model = SmallCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(train_scratch.CHECKPOINT_PATH))
    return train_scratch.predict(model, images[split["test"]], mean, std)


def plot_confusion(cm, class_names, path, model_name):
    # Color by share of the true class, so a 3% error reads the same in any class size
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    # Power scale so a 2-5% error cell is visibly tinted next to a 95% diagonal
    ax.imshow(share, cmap="Blues", norm=PowerNorm(gamma=0.4, vmin=0, vmax=1))

    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j] == 0:
                continue
            color = "white" if share[i, j] > 0.3 else "#2b2b2b"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9, color=color)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(ticks, class_names, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=10, color="#555555")
    ax.set_ylabel("True class", fontsize=10, color="#555555")
    # White gaps between cells
    ax.set_xticks(ticks - 0.5, minor=True)
    ax.set_yticks(ticks - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"{model_name}, test split. Counts; shade is share of the true class",
                 fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    val_accuracies = {}
    for name, path in [("small_cnn", train_scratch.RESULTS_PATH),
                       ("resnet18", train_resnet.RESULTS_PATH)]:
        with open(path) as f:
            val_accuracies[name] = json.load(f)["best_val_accuracy"]
        print(f"{name:10s} val accuracy {val_accuracies[name]:.4f}")

    # Decided on validation alone, before any test prediction exists
    chosen = choose_model(val_accuracies)
    print(f"\nChosen on validation: {chosen}")

    dataset = load_eurosat()
    class_names = dataset.classes
    images = torch.from_numpy(load_images(dataset))
    split = load_split(SPLIT_PATH, dataset.targets)
    test_y = np.array(dataset.targets)[split["test"]]

    test_preds = predict_test(chosen, images, split, len(class_names)).numpy()
    n = len(test_y)
    acc = float((test_preds == test_y).mean())
    # Normal approximation to the binomial; fine at n = 4050
    half_width = 1.96 * math.sqrt(acc * (1 - acc) / n)
    print(f"Test accuracy {acc:.4f}  (95% CI {acc - half_width:.4f} to {acc + half_width:.4f}, n={n})")

    cm = confusion_matrix(test_y, test_preds, labels=list(range(len(class_names))))
    per_class = {name: float(cm[c, c] / cm[c].sum()) for c, name in enumerate(class_names)}

    confusions = top_confusions(cm, class_names)
    print("\nLargest confusions (true -> predicted):")
    for true, pred, count in confusions:
        print(f"  {true:22s} -> {pred:22s} {count}")

    idx = {name: c for c, name in enumerate(class_names)}
    pairs = {}
    print("\nPairs named in advance:")
    for a, b in PAIRS_OF_INTEREST:
        a_as_b, b_as_a = int(cm[idx[a], idx[b]]), int(cm[idx[b], idx[a]])
        pairs[f"{a} / {b}"] = {f"{a} -> {b}": a_as_b, f"{b} -> {a}": b_as_a}
        print(f"  {a} -> {b}: {a_as_b}   {b} -> {a}: {b_as_a}")

    plot_confusion(cm, class_names, FIGURE_PATH, chosen)
    print(f"\nSaved {FIGURE_PATH}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "chosen_model": chosen,
            "val_accuracy": val_accuracies,
            "test_accuracy": acc,
            "test_accuracy_95ci": [acc - half_width, acc + half_width],
            "n_test": n,
            "test_accuracy_by_class": per_class,
            "largest_confusions": [{"true": t, "predicted": p, "count": c} for t, p, c in confusions],
            "pairs_of_interest": pairs,
            "class_names": class_names,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)


if __name__ == "__main__":
    main()
