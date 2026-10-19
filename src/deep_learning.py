"""
Stage 3 - A basic feed-forward neural network in PyTorch.

Uses the same train/test split as the classical models (from Phase 1).
10% of the TRAINING set is held out as a validation set for early stopping,
so the test set is never used during training.

Network:  input -> 64 -> ReLU -> Dropout -> 32 -> ReLU -> Dropout -> one output per class
Loss:     cross-entropy (with class weights if Phase 1 chose class weights)
Optimiser: Adam

Run from the project root (after Phase 1):
    python -m src.deep_learning
    python -m src.deep_learning --extra-test data/nsl-kdd/KDDTest+.txt   # NSL-KDD only
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import compare_models, evaluate_model, load_extra_test, save_curves
from src.preprocessing import RANDOM_STATE, load_processed_data


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=(64, 32), dropout=0.2):
        super().__init__()
        layers, size = [], input_dim
        for h in hidden:
            layers += [nn.Linear(size, h), nn.ReLU(), nn.Dropout(dropout)]
            size = h
        layers.append(nn.Linear(size, n_classes))  # raw scores (logits), one per class
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def predict(model, X, batch_size=4096):
    """Return the predicted class index for every row of X (numpy array)."""
    model.eval()  # turns dropout off
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(X[i:i + batch_size]))
            outputs.append(logits.argmax(dim=1).numpy())
    return np.concatenate(outputs)


def train_network(X_train, y_train, n_classes, class_weights=None, epochs=100, batch_size=256,
                  lr=1e-3, patience=5):
    """Train with early stopping on a validation split; returns the best model and its history."""
    torch.manual_seed(RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE))
    X_val_t, y_val_t = torch.from_numpy(X_val), torch.from_numpy(y_val).long()

    model = FeedForwardNet(X_train.shape[1], n_classes)
    weight = None
    if class_weights:
        weight = torch.tensor([class_weights[str(c)] for c in range(n_classes)], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        train_loss = total / len(y_tr)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"  epoch {epoch:2d}: train loss {train_loss:.4f}, "
              f"val loss {val_loss:.4f}, val accuracy {val_acc:.4f}")

        # early stopping: keep the weights with the lowest validation loss
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  no improvement for {patience} epochs, stopping")
                break

    model.load_state_dict(best_state)
    return model, history


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(epochs, history["val_accuracy"], color="tab:green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Validation accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_network(path):
    """Rebuild a saved network (used later by the Streamlit app)."""
    checkpoint = torch.load(path, weights_only=True)
    model = FeedForwardNet(checkpoint["input_dim"], checkpoint["n_classes"],
                           tuple(checkpoint["hidden"]), checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def run(artifacts_dir="artifacts", epochs=100, batch_size=256, lr=1e-3, extra_test=None):
    out = Path(artifacts_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, meta = load_processed_data(artifacts_dir)
    n_classes = len(meta["class_names"])
    print(f"Train: {X_train.shape}, test: {X_test.shape}, classes: {meta['class_names']}")
    extra = load_extra_test(extra_test, meta, artifacts_dir) if extra_test else None

    print("\n=== neural_network ===")
    start = time.time()
    model, history = train_network(X_train, y_train, n_classes, meta["class_weights"],
                                   epochs=epochs, batch_size=batch_size, lr=lr)
    train_info = {"train_seconds": round(time.time() - start, 2), "train_rows": int(len(y_train)),
                  "epochs_run": len(history["train_loss"]),
                  "history": {k: [round(v, 5) for v in vals] for k, vals in history.items()}}
    plot_history(history, out / "plots" / "nn_training_curve.png")

    evaluate_model("neural_network", lambda X: predict(model, X), X_test, y_test, meta,
                   artifacts_dir, extra, train_info)
    torch.save({"state_dict": model.state_dict(), "input_dim": X_train.shape[1],
                "n_classes": n_classes, "hidden": [64, 32], "dropout": 0.2},
               out / "models" / "neural_network.pt")

    compare_models(artifacts_dir)
    save_curves(artifacts_dir)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: PyTorch feed-forward network")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--epochs", type=int, default=100, help="maximum epochs (early stopping may end sooner)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--extra-test", help="optional second test file, e.g. KDDTest+.txt")
    args = parser.parse_args()
    run(args.artifacts, args.epochs, args.batch_size, args.lr, args.extra_test)


if __name__ == "__main__":
    main()
