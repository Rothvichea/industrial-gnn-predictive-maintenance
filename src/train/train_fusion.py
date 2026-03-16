import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.data.cwru_dataset_v2 import CWRUDatasetV2, LABEL_NAMES
from src.data.graph_builder import build_graph, NODES, FAULT_TYPE_MAP
from src.models.fusion_model import FusionModel

CFG = {
    "model":           "FusionModel_CNN_GNN",
    "cnn_embed_dim":   128,
    "gnn_embed_dim":   128,
    "dropout":         0.4,
    "batch_size":      64,
    "lr":              1e-3,
    "weight_decay":    3e-4,
    "epochs":          60,
    "label_smoothing": 0.1,
    "patience":        55,
    "window_size":     1024,
    "stride":          512,
    "seed":            42,
    "num_classes":     4,
}

# ── Node ID mapping ────────────────────────────────────────────────────────
# Map each (fault_type, severity, load) → node_id in the graph
NODE_LOOKUP = {
    (FAULT_TYPE_MAP[ft], sev, load): nid
    for nid, ft, sev, load, rpm in NODES
}

def get_node_id(label: int, severity: int, load: int) -> int:
    key = (label, severity, load)
    return NODE_LOOKUP.get(key, 0)


class CWRUFusionDataset(Dataset):
    """
    Wraps CWRUDatasetV2 and adds node_id per sample
    so the fusion model can look up the GNN embedding.
    """
    def __init__(self, split, window_size=1024, stride=512):
        from src.data.cwru_dataset_v2 import CWRUDatasetV2, CWRU_MANIFEST
        from pathlib import Path
        import numpy as np
        import scipy.io as sio

        root = Path("data/raw/CWRU")
        from collections import defaultdict
        by_label = defaultdict(list)
        for entry in CWRU_MANIFEST:
            by_label[entry[1]].append(entry)

        rng = np.random.default_rng(CFG["seed"])
        selected = []
        for label, entries in by_label.items():
            entries = list(entries)
            rng.shuffle(entries)
            n    = len(entries)
            n_tr = max(1, int(n * 0.6))
            n_val= max(1, int(n * 0.2))
            if split == "train":
                selected.extend(entries[:n_tr])
            elif split == "val":
                selected.extend(entries[n_tr:n_tr+n_val])
            else:
                selected.extend(entries[n_tr+n_val:])

        all_wins, all_labels, all_node_ids = [], [], []

        for rel_path, label, fault_type, severity in selected:
            full_path = root / rel_path
            if not full_path.exists():
                continue
            mat    = sio.loadmat(str(full_path))
            signal = None
            for k in mat.keys():
                if not k.startswith('_') and 'DE_time' in k:
                    signal = mat[k].flatten().astype('float32')
                    break
            if signal is None:
                continue

            starts = range(0, len(signal) - window_size + 1, stride)
            wins   = np.array([signal[s:s+window_size] for s in starts], dtype='float32')
            mean   = wins.mean(axis=1, keepdims=True)
            std    = wins.std(axis=1,  keepdims=True) + 1e-8
            wins   = (wins - mean) / std

            # Assign node_id — use load=0 as default (most files are load 0)
            node_id = get_node_id(label, severity, 0)
            node_ids = np.full(len(wins), node_id, dtype='int64')

            all_wins.append(wins)
            all_labels.append(np.full(len(wins), label, dtype='int64'))
            all_node_ids.append(node_ids)

        X        = np.concatenate(all_wins)
        y        = np.concatenate(all_labels)
        node_ids = np.concatenate(all_node_ids)

        idx = rng.permutation(len(X))
        self.X        = torch.from_numpy(X[idx]).unsqueeze(1)
        self.y        = torch.from_numpy(y[idx])
        self.node_ids = torch.from_numpy(node_ids[idx])

        unique, counts = torch.unique(self.y, return_counts=True)
        print(f"[FusionDataset] split='{split}' total={len(self.X)}")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"  label {u} ({LABEL_NAMES[u]}): {c} windows")

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.node_ids[idx]


def get_class_weights(dataset):
    counts = torch.zeros(CFG["num_classes"])
    for _, y, _ in dataset:
        counts[y.item()] += 1
    w = 1.0 / (counts + 1e-8)
    return w / w.sum() * CFG["num_classes"]


def run_epoch(model, loader, graph_x, edge_index, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, node_ids in tqdm(loader, desc="  train" if train else "  eval", leave=False):
            x, y, node_ids = x.to(device), y.to(device), node_ids.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x, node_ids, graph_x, edge_index)
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

    # Graph — stays fixed on GPU throughout training
    graph      = build_graph()
    graph_x    = graph.x.to(device)
    edge_index = graph.edge_index.to(device)

    train_ds = CWRUFusionDataset("train")
    val_ds   = CWRUFusionDataset("val")
    test_ds  = CWRUFusionDataset("test")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model     = FusionModel(
        cnn_embed_dim=CFG["cnn_embed_dim"],
        gnn_embed_dim=CFG["gnn_embed_dim"],
        num_classes=CFG["num_classes"],
        dropout=CFG["dropout"],
    ).to(device)

    weights   = get_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=CFG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    label_names  = list(LABEL_NAMES.values())
    best_val_acc = 0.0
    patience_ctr = 0
    best_path    = "experiments/best_fusion.pt"
    os.makedirs("experiments", exist_ok=True)

    mlflow.set_experiment("cwru_fault_detection")

    with mlflow.start_run(run_name=CFG["model"]):
        mlflow.log_params(CFG)

        for epoch in range(1, CFG["epochs"]+1):
            tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, graph_x, edge_index, optimizer, criterion, device, train=True)
            va_loss, va_acc, _, _ = run_epoch(model, val_loader,   graph_x, edge_index, optimizer, criterion, device, train=False)
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
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print("\n=== Test Evaluation (best fusion model) ===")
        model.load_state_dict(torch.load(best_path))
        te_loss, te_acc, preds, labels = run_epoch(
            model, test_loader, graph_x, edge_index,
            optimizer, criterion, device, train=False
        )
        report = classification_report(labels, preds, target_names=label_names)
        print(f"Test loss={te_loss:.4f}  Test acc={te_acc:.4f}")
        print(report)

        mlflow.log_metrics({"test_loss": te_loss, "test_acc": te_acc})
        mlflow.log_text(report, "fusion_test_report.txt")
        mlflow.pytorch.log_model(model, "model")

        print(f"\n{'='*50}")
        print(f"ABLATION SUMMARY")
        print(f"  CNN only (baseline): 98.49%")
        print(f"  CNN + GNN (fusion):  {te_acc*100:.2f}%")
        print(f"  Delta:               {(te_acc - 0.9849)*100:+.2f}%")
        print(f"{'='*50}")


if __name__ == "__main__":
    train()
