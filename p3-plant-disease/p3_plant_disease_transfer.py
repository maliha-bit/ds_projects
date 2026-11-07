"""
Project 3: Image classification with transfer learning (ResNet18) - plant disease.

Dataset: PlantVillage (Kaggle). Arrange as one folder per class:
    data/plantvillage/
        Tomato_healthy/*.jpg
        Tomato_Early_blight/*.jpg
        ...

Run:
    python p3_plant_disease_transfer.py --data_dir data/plantvillage --epochs 5

Tip: use --max_per_class 300 to train fast on a laptop/CPU.
Outputs go to ./outputs_p3/ : model, classification report, confusion matrix, misclassified list.
"""
import argparse
import json
import os
import random

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs_p3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_splits(targets, max_per_class, val_frac):
    rng = random.Random(SEED)
    by_class = {}
    for idx, t in enumerate(targets):
        by_class.setdefault(t, []).append(idx)
    train_idx, val_idx = [], []
    for t, idxs in by_class.items():
        rng.shuffle(idxs)
        if max_per_class:
            idxs = idxs[:max_per_class]
        n_val = max(1, int(len(idxs) * val_frac))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--finetune", action="store_true", help="unfreeze last block after head training")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    val_tf = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )

    train_full = datasets.ImageFolder(args.data_dir, transform=train_tf)
    val_full = datasets.ImageFolder(args.data_dir, transform=val_tf)
    classes = train_full.classes
    print(f"{len(classes)} classes, {len(train_full)} images")

    train_idx, val_idx = make_splits(train_full.targets, args.max_per_class, 0.2)
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=args.batch_size, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(DEVICE)

    def run_epochs(n_epochs, lr, tag):
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        for ep in range(1, n_epochs + 1):
            model.train()
            total, correct, loss_sum = 0, 0, 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out = model(x)
                loss = nn.functional.cross_entropy(out, y)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            val_acc = evaluate(model, val_loader)[0]
            print(f"[{tag}] epoch {ep}/{n_epochs} loss={loss_sum/total:.4f} train_acc={correct/total:.4f} val_acc={val_acc:.4f}")

    run_epochs(args.epochs, args.lr, "head")

    if args.finetune:
        for p in model.layer4.parameters():
            p.requires_grad = True
        run_epochs(max(2, args.epochs // 2), args.lr / 10, "finetune")

    acc, preds, labels = evaluate(model, val_loader)
    print(f"\nFinal validation accuracy: {acc:.4f}")

    report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    with open(f"{OUT}/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
    plt.close()

    # Error analysis: list misclassified validation images
    wrong = []
    for i, (p, t) in enumerate(zip(preds, labels)):
        if p != t:
            path = val_full.samples[val_idx[i]][0]
            wrong.append({"file": path, "true": classes[t], "predicted": classes[p]})
    pd.DataFrame(wrong).to_csv(f"{OUT}/misclassified.csv", index=False)
    print(f"{len(wrong)} misclassified images listed in {OUT}/misclassified.csv")

    torch.save({"state_dict": model.state_dict(), "classes": classes}, f"{OUT}/resnet18_plant.pt")


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE))
            preds.append(out.argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return float((preds == labels).mean()), preds, labels


if __name__ == "__main__":
    main()
