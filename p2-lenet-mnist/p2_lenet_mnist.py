"""
Project 2: Small paper reproduction - LeNet-5 (LeCun et al., 1998) on MNIST,
compared against a modern small CNN (ReLU + MaxPool + Dropout).

    python p2_lenet_mnist.py --epochs 10

MNIST downloads automatically. Outputs go to ./outputs_p2/
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs_p2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LeNet5(nn.Module):
    """Classic LeNet-5: tanh + average pooling, 32x32 input."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.f1 = nn.Linear(16 * 5 * 5, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(torch.tanh(self.c1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.c2(x)), 2)
        x = x.flatten(1)
        x = torch.tanh(self.f1(x))
        x = torch.tanh(self.f2(x))
        return self.f3(x)


class ModernCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.drop = nn.Dropout(0.3)
        self.f1 = nn.Linear(64 * 8 * 8, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = self.drop(x.flatten(1))
        x = F.relu(self.f1(x))
        return self.f2(x)


def get_loaders(batch_size):
    tf = transforms.Compose(
        [transforms.Pad(2), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST("data", train=True, download=True, transform=tf)
    test = datasets.MNIST("data", train=False, download=True, transform=tf)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=1000),
    )


def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            p = out.argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
            preds.append(p.cpu())
            labels.append(y.cpu())
    return loss_sum / total, correct / total, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def train_model(name, model, train_loader, test_loader, epochs, lr):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "test_loss": [], "test_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        tr_loss = running / len(train_loader.dataset)
        te_loss, te_acc, _, _ = evaluate(model, test_loader)
        hist["train_loss"].append(tr_loss)
        hist["test_loss"].append(te_loss)
        hist["test_acc"].append(te_acc)
        print(f"[{name}] epoch {ep}/{epochs} train_loss={tr_loss:.4f} test_acc={te_acc:.4f}")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(42)
    train_loader, test_loader = get_loaders(args.batch_size)

    results, histories = {}, {}
    for name, cls in [("LeNet5", LeNet5), ("ModernCNN", ModernCNN)]:
        model = cls()
        n_params = sum(p.numel() for p in model.parameters())
        histories[name] = train_model(name, model, train_loader, test_loader, args.epochs, args.lr)
        _, acc, preds, labels = evaluate(model, test_loader)
        results[name] = {"params": n_params, "final_test_acc": round(acc, 4)}

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"{name} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{OUT}/{name}_confusion.png", dpi=150)
        plt.close()
        torch.save(model.state_dict(), f"{OUT}/{name}.pt")

    plt.figure(figsize=(6, 4))
    for name, h in histories.items():
        plt.plot(h["test_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.title("LeNet-5 vs Modern CNN on MNIST")
    plt.tight_layout()
    plt.savefig(f"{OUT}/accuracy_curves.png", dpi=150)
    plt.close()

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
