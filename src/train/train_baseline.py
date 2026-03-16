"""
Training loop for the baseline FaultClassifier (no GNN).
This is Ablation A — CNN-only, no graph structure.
Results logged to MLflow for comparison against the full GNN model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.data.cwru_dataset_v2 import CWRUDatasetV2 as CWRUDataset
from src.models.temporal_encoder import FaultClassifier


# ── Config ────────────────────────────────────────────────────────────────
CFG = {
    "model":        "FaultClassifier_CNN_baseline",
    "embed_dim":    128,
    "dropout":      0.3,
    "batch_size":   64,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       30,
    "window_size":  1024,
    "stride":       512,
    "seed":         42,
    "num_classes":  4,
}


def get_class_weights(dataset: CWRUDataset) -> torch.Tensor:
    """Inverse-frequency weights to handle class imbalance."""
    counts = torch.zeros(CFG["num_classes"])
    for _, y in dataset:
        counts[y.item()] += 1
    weights = 1.0 / (counts + 1e-8)
    return weights / weights.sum() * CFG["num_classes"]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds       = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def train():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = CWRUDataset(split="train", window_size=CFG["window_size"], stride=CFG["stride"])
    val_ds   = CWRUDataset(split="val",   window_size=CFG["window_size"], stride=CFG["stride"])
    test_ds  = CWRUDataset(split="test",  window_size=CFG["window_size"], stride=CFG["stride"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = FaultClassifier(
        num_classes=CFG["num_classes"],
        embed_dim=CFG["embed_dim"],
        dropout=CFG["dropout"],
    ).to(device)

    # Weighted loss for class imbalance
    weights   = get_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    label_names = list(train_ds.label_names.values())

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_experiment("cwru_fault_detection")

    with mlflow.start_run(run_name=CFG["model"]):
        mlflow.log_params(CFG)

        best_val_acc = 0.0
        best_model_path = "experiments/best_baseline.pt"

        for epoch in range(1, CFG["epochs"] + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss":   va_loss, "val_acc":   va_acc,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            print(f"Epoch {epoch:02d}/{CFG['epochs']}  "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

            # Save best model
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"  → New best val_acc={va_acc:.4f} saved")

        # ── Final test evaluation ──────────────────────────────────────────
        print("\n=== Test Evaluation (best model) ===")
        model.load_state_dict(torch.load(best_model_path))
        te_loss, te_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        print(f"Test loss={te_loss:.4f}  Test acc={te_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=label_names))

        mlflow.log_metrics({"test_loss": te_loss, "test_acc": te_acc})
        mlflow.log_text(
            classification_report(labels, preds, target_names=label_names),
            "test_classification_report.txt"
        )
        mlflow.pytorch.log_model(model, "model")

        print(f"\nBest val_acc: {best_val_acc:.4f}")
        print(f"MLflow run logged. View with: mlflow ui --port 5000")


if __name__ == "__main__":
    train()
