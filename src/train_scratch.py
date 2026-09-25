"""Train SmallCNN from scratch. Reports validation accuracy only, never test."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import load_eurosat, load_images
from model import SmallCNN
from split import SEED, SPLIT_PATH, load_split

EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
RESULTS_PATH = Path("results/small_cnn.json")
CHECKPOINT_PATH = Path("models/small_cnn.pt")


def channel_stats(train_images):
    """Per-channel mean and std of the train images, shaped to broadcast over a batch."""
    x = train_images.float() / 255.0
    return x.mean(dim=(0, 2, 3)).view(1, 3, 1, 1), x.std(dim=(0, 2, 3)).view(1, 3, 1, 1)


def to_float(batch, mean, std):
    return (batch.float() / 255.0 - mean) / std


def random_flips(x):
    # Overhead imagery has no fixed up or left
    flip_h = torch.rand(len(x)) < 0.5
    flip_v = torch.rand(len(x)) < 0.5
    x[flip_h] = x[flip_h].flip(3)
    x[flip_v] = x[flip_v].flip(2)
    return x


def predict(model, images, mean, std):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(images), 512):
            batch = to_float(images[start:start + 512], mean, std)
            preds.append(model(batch).argmax(dim=1))
    return torch.cat(preds)


def train_one_epoch(model, optimizer, images, labels, mean, std):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    order = torch.randperm(len(images))
    total_loss, correct = 0.0, 0

    for start in range(0, len(images), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        x = random_flips(to_float(images[idx], mean, std))
        y = labels[idx]

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
        correct += (out.argmax(dim=1) == y).sum().item()

    return total_loss / len(images), correct / len(images)


def main():
    torch.manual_seed(SEED)
    dataset = load_eurosat()
    images = torch.from_numpy(load_images(dataset))
    labels = torch.tensor(dataset.targets)
    split = load_split(SPLIT_PATH, dataset.targets)

    train_x, train_y = images[split["train"]], labels[split["train"]]
    val_x, val_y = images[split["val"]], labels[split["val"]]

    # Channel stats from train only, so val leaks nothing
    mean, std = channel_stats(train_x)

    model = SmallCNN(num_classes=len(dataset.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val, best_epoch = 0.0, 0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_x, train_y, mean, std)
        val_acc = (predict(model, val_x, mean, std) == val_y).float().mean().item()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_accuracy": train_acc, "val_accuracy": val_acc})
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  train {train_acc:.4f}  val {val_acc:.4f}")

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    minutes = (time.time() - start_time) / 60
    print(f"\nBest val accuracy {best_val:.4f} at epoch {best_epoch} ({minutes:.1f} min)")

    # Per class val accuracy at the best epoch
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    val_preds = predict(model, val_x, mean, std)
    per_class = {}
    for c, name in enumerate(dataset.classes):
        mask = val_y == c
        per_class[name] = (val_preds[mask] == c).float().mean().item()
        print(f"  {name:22s} {per_class[name]:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val,
            "final_val_accuracy": history[-1]["val_accuracy"],
            "val_accuracy_by_class": per_class,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "train_minutes": round(minutes, 1),
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
