"""
Baseline v2 — stronger regularization to fix overfitting.
Changes vs v1:
  - dropout 0.3 → 0.5
  - label smoothing 0.1
  - weight decay 1e-4 → 3e-4
  - early stopping (patience=)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.data.cwru_dataset_v2 import CWRUDatasetV2 as CWRUDataset
from src.models.temporal_encoder import FaultClassifier

CFG = {
    "model":          "FaultClassifier_CNN_v2_regularized",
    "embed_dim":      128,
    "dropout":        0.5,
    "batch_size":     64,
    "lr":             1e-3,
    "weight_decay":   3e-4,
    "epochs":         60,
    "label_smoothing":0.1,
    "patience":       50,
    "window_size":    1024,
    "stride":         512,
    "seed":           42,
    "num_classes":    4,
}


def get_class_weights(dataset):
    import torch
    counts = torch.zeros(CFG["num_classes"])
    for _, y in dataset:
        counts[y.item()] += 1
    w = 1.0 / (counts + 1e-8)
    return w / w.sum() * CFG["num_classes"]


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc="  train" if train else "  eval", leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds       = logits.argmax(1)
            correct    += (preds == y).sum().item()
            total      += len(y)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return total_loss/total, correct/total, all_preds, all_labels


def train():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = CWRUDataset(split="train", window_size=CFG["window_size"], stride=CFG["stride"])
    val_ds   = CWRUDataset(split="val",   window_size=CFG["window_size"], stride=CFG["stride"])
    test_ds  = CWRUDataset(split="test",  window_size=CFG["window_size"], stride=CFG["stride"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model = FaultClassifier(
        num_classes=CFG["num_classes"],
        embed_dim=CFG["embed_dim"],
        dropout=CFG["dropout"],
    ).to(device)

    weights   = get_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=CFG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    label_names  = list(train_ds.label_names.values())
    best_val_acc = 0.0
    patience_ctr = 0
    best_path    = "experiments/best_baseline_v2.pt"

    mlflow.set_experiment("cwru_fault_detection")

    with mlflow.start_run(run_name=CFG["model"]):
        mlflow.log_params(CFG)

        for epoch in range(1, CFG["epochs"]+1):
            tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
            va_loss, va_acc, _, _ = run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss":   va_loss, "val_acc":   va_acc,
            }, step=epoch)

            print(f"Epoch {epoch:02d}/{CFG['epochs']}  "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
                  f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                patience_ctr = 0
                torch.save(model.state_dict(), best_path)
                print(f"  → best val_acc={va_acc:.4f} saved")
            else:
                patience_ctr += 1
                if patience_ctr >= CFG["patience"]:
                    print(f"\nEarly stopping at epoch {epoch} (patience={CFG['patience']})")
                    break

        print("\n=== Test Evaluation ===")
        model.load_state_dict(torch.load(best_path))
        te_loss, te_acc, preds, labels = run_epoch(model, test_loader, optimizer, criterion, device, train=False)

        report = classification_report(labels, preds, target_names=label_names)
        print(f"Test loss={te_loss:.4f}  Test acc={te_acc:.4f}")
        print(report)

        mlflow.log_metrics({"test_loss": te_loss, "test_acc": te_acc})
        mlflow.log_text(report, "test_report.txt")
        mlflow.pytorch.log_model(model, "model")
        print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
