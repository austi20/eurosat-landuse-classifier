"""Fine-tune an ImageNet pretrained ResNet18. Reports validation accuracy only, never test."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_eurosat, load_images
from model import build_resnet18
from split import SEED, SPLIT_PATH, load_split
from train_scratch import random_flips

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
RESULTS_PATH = Path("results/resnet18.json")
CHECKPOINT_PATH = Path("models/resnet18.pt")

# The statistics the pretrained weights were trained with, not EuroSAT's
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare(batch):
    """uint8 (N, 3, 64, 64) -> normalized float (N, 3, 224, 224), channels last."""
    x = batch.float() / 255.0
    x = F.interpolate(x, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    # Roughly 25% faster convolutions on CPU
    return x.contiguous(memory_format=torch.channels_last)


def predict(model, images):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(images), 256):
            preds.append(model(prepare(images[start:start + 256])).argmax(dim=1))
    return torch.cat(preds)


def train_one_epoch(model, optimizer, images, labels):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    order = torch.randperm(len(images))
    total_loss, correct = 0.0, 0
    start_time = time.time()

    for step, start in enumerate(range(0, len(images), BATCH_SIZE), 1):
        idx = order[start:start + BATCH_SIZE]
        x = random_flips(prepare(images[idx]))
        y = labels[idx]

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
        correct += (out.argmax(dim=1) == y).sum().item()
        # An epoch is about half an hour on a laptop CPU, so show progress
        if step % 50 == 0:
            seen = start + len(idx)
            print(f"  {seen:5d}/{len(images)}  loss {total_loss / seen:.4f}  "
                  f"{(time.time() - start_time) / 60:.1f} min", flush=True)

    return total_loss / len(images), correct / len(images)


def main():
    torch.manual_seed(SEED)
    dataset = load_eurosat()
    images = torch.from_numpy(load_images(dataset))
    labels = torch.tensor(dataset.targets)
    split = load_split(SPLIT_PATH, dataset.targets)

    train_x, train_y = images[split["train"]], labels[split["train"]]
    val_x, val_y = images[split["val"]], labels[split["val"]]

    model = build_resnet18(num_classes=len(dataset.classes))
    model = model.to(memory_format=torch.channels_last)
    # Every layer is fine-tuned, not just the new head
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val, best_epoch = 0.0, 0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_x, train_y)
        val_acc = (predict(model, val_x) == val_y).float().mean().item()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_accuracy": train_acc, "val_accuracy": val_acc})
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  train {train_acc:.4f}  val {val_acc:.4f}",
              flush=True)

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    minutes = (time.time() - start_time) / 60
    print(f"\nBest val accuracy {best_val:.4f} at epoch {best_epoch} ({minutes:.1f} min)")

    # Per class val accuracy at the best epoch
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    val_preds = predict(model, val_x)
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
            "image_size": IMAGE_SIZE,
            "seed": SEED,
            "train_minutes": round(minutes, 1),
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
