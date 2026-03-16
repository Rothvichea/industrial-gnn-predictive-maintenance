import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from src.data.cwru_dataset_v2 import CWRUDatasetV2, LABEL_NAMES
from src.data.graph_builder import build_graph, NODES, FAULT_TYPE_MAP
from src.models.temporal_encoder import FaultClassifier
from src.models.fusion_model import FusionModel

# ── Config ────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_LIST  = list(LABEL_NAMES.values())
COLORS      = ["#1D9E75", "#D85A30", "#7F77DD", "#EF9F27"]
DARK_BG     = "#0F0F13"
PANEL_BG    = "#1A1A22"
TEXT_COLOR  = "#D3D1C7"
MUTED       = "#888780"
GRID_COLOR  = "#2A2A32"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

NODE_LOOKUP = {
    (FAULT_TYPE_MAP[ft], sev, load): nid
    for nid, ft, sev, load, rpm in NODES
}

# ── Data helpers ──────────────────────────────────────────────────────────
class FusionTestDataset(torch.utils.data.Dataset):
    def __init__(self):
        from src.data.cwru_dataset_v2 import CWRU_MANIFEST
        from pathlib import Path
        import scipy.io as sio
        from collections import defaultdict

        root = Path("data/raw/CWRU")
        by_label = defaultdict(list)
        for entry in CWRU_MANIFEST:
            by_label[entry[1]].append(entry)

        rng = np.random.default_rng(42)
        selected = []
        for label, entries in by_label.items():
            entries = list(entries)
            rng.shuffle(entries)
            n = len(entries)
            n_tr  = max(1, int(n * 0.6))
            n_val = max(1, int(n * 0.2))
            selected.extend(entries[n_tr+n_val:])

        all_wins, all_labels, all_node_ids = [], [], []
        for rel_path, label, fault_type, severity in selected:
            full_path = root / rel_path
            if not full_path.exists():
                continue
            mat = sio.loadmat(str(full_path))
            signal = None
            for k in mat.keys():
                if not k.startswith('_') and 'DE_time' in k:
                    signal = mat[k].flatten().astype('float32')
                    break
            if signal is None:
                continue
            starts = range(0, len(signal) - 1024 + 1, 512)
            wins   = np.array([signal[s:s+1024] for s in starts], dtype='float32')
            mean   = wins.mean(axis=1, keepdims=True)
            std    = wins.std(axis=1, keepdims=True) + 1e-8
            wins   = (wins - mean) / std
            node_id = NODE_LOOKUP.get((label, severity, 0), 0)
            all_wins.append(wins)
            all_labels.append(np.full(len(wins), label, dtype='int64'))
            all_node_ids.append(np.full(len(wins), node_id, dtype='int64'))

        idx = rng.permutation(sum(len(w) for w in all_wins))
        X        = np.concatenate(all_wins)[idx]
        y        = np.concatenate(all_labels)[idx]
        node_ids = np.concatenate(all_node_ids)[idx]
        self.X        = torch.from_numpy(X).unsqueeze(1)
        self.y        = torch.from_numpy(y)
        self.node_ids = torch.from_numpy(node_ids)

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.node_ids[idx]


@torch.no_grad()
def get_predictions(model, loader, graph_x, edge_index, has_node_ids=True):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        if has_node_ids:
            x, y, node_ids = batch
            x, y, node_ids = x.to(DEVICE), y.to(DEVICE), node_ids.to(DEVICE)
            logits = model(x, node_ids, graph_x, edge_index)
        else:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Plot functions ────────────────────────────────────────────────────────
def plot_confusion_matrix(ax, labels, preds, title):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(["Normal", "Inner\nRace", "Ball", "Outer\nRace"], fontsize=9)
    ax.set_yticklabels(["Normal", "Inner\nRace", "Ball", "Outer\nRace"], fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=10)

    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else TEXT_COLOR
            ax.text(j, i, f"{val:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold")
    return im


def plot_roc_curves(ax, labels, probs, title):
    labels_bin = label_binarize(labels, classes=[0,1,2,3])
    for i, (name, color) in enumerate(zip(LABEL_LIST, COLORS)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1], color=MUTED, lw=1, linestyle="--", alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, pad=10)
    ax.legend(fontsize=8, loc="lower right",
              facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(True)


def plot_per_class_metrics(ax, labels, preds, title):
    report = classification_report(labels, preds,
                                   target_names=LABEL_LIST,
                                   output_dict=True)
    metrics    = ["precision", "recall", "f1-score"]
    x          = np.arange(4)
    width      = 0.25
    metric_colors = ["#1D9E75", "#7F77DD", "#EF9F27"]

    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        vals = [report[cls][metric] for cls in LABEL_LIST]
        bars = ax.bar(x + i*width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.85, edgecolor=DARK_BG, linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x + width)
    ax.set_xticklabels(LABEL_LIST, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title, pad=10)
    ax.legend(fontsize=9, facecolor=PANEL_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(True, axis="y")


def plot_confidence_distribution(ax, labels, probs, title):
    correct_mask   = (probs.argmax(axis=1) == labels)
    correct_conf   = probs.max(axis=1)[correct_mask]
    incorrect_conf = probs.max(axis=1)[~correct_mask]

    bins = np.linspace(0, 1, 25)
    ax.hist(correct_conf,   bins=bins, color="#1D9E75", alpha=0.75,
            label=f"Correct ({correct_mask.sum()})", edgecolor=DARK_BG, linewidth=0.4)
    ax.hist(incorrect_conf, bins=bins, color="#D85A30", alpha=0.75,
            label=f"Wrong ({(~correct_mask).sum()})", edgecolor=DARK_BG, linewidth=0.4)
    ax.set_xlabel("Confidence (max softmax prob)")
    ax.set_ylabel("Count")
    ax.set_title(title, pad=10)
    ax.legend(fontsize=9, facecolor=PANEL_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(True, axis="y")


def plot_ablation_bar(ax):
    models   = ["CNN only\n(baseline)", "CNN + GNN\n(fusion)"]
    accs     = [98.49, 99.10]
    bar_colors = [COLORS[2], COLORS[0]]
    bars = ax.bar(models, accs, color=bar_colors, alpha=0.85,
                  edgecolor=DARK_BG, linewidth=0.5, width=0.4)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.2f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylim(97.5, 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Ablation study", pad=10)
    ax.grid(True, axis="y")

    # Delta annotation
    ax.annotate("", xy=(1, 99.10), xytext=(0, 98.49),
                arrowprops=dict(arrowstyle="->", color="#EF9F27", lw=2))
    ax.text(0.5, 98.82, "+0.61%", ha="center", fontsize=11,
            color="#EF9F27", fontweight="bold")


def plot_signal_samples(ax_row, test_ds):
    """Plot one sample signal per class."""
    shown = {}
    for x, y, _ in test_ds:
        label = y.item()
        if label not in shown:
            shown[label] = x.squeeze().numpy()
        if len(shown) == 4:
            break

    for i, (label, signal) in enumerate(sorted(shown.items())):
        ax = ax_row[i]
        ax.plot(signal, color=COLORS[label], linewidth=0.8, alpha=0.9)
        ax.set_title(f"{LABEL_LIST[label]}", pad=6, color=COLORS[label])
        ax.set_xlabel("Sample", fontsize=9)
        if i == 0:
            ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True)
        ax.set_xlim(0, 1024)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    os.makedirs("experiments/plots", exist_ok=True)

    # Load graph
    graph      = build_graph()
    graph_x    = graph.x.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)

    # Load baseline model
    baseline = FaultClassifier(num_classes=4, embed_dim=128).to(DEVICE)
    baseline.load_state_dict(torch.load("experiments/best_baseline_v2.pt",
                                         map_location=DEVICE))

    # Load fusion model
    fusion = FusionModel(num_classes=4).to(DEVICE)
    fusion.load_state_dict(torch.load("experiments/best_fusion.pt",
                                       map_location=DEVICE))

    # Test datasets
    base_test = CWRUDatasetV2(split="test")
    fuse_test = FusionTestDataset()

    base_loader = DataLoader(base_test, batch_size=128, shuffle=False, num_workers=2)
    fuse_loader = DataLoader(fuse_test, batch_size=128, shuffle=False, num_workers=2)

    print("Running inference...")
    base_labels, base_preds, base_probs = get_predictions(
        baseline, base_loader, graph_x, edge_index, has_node_ids=False)
    fuse_labels, fuse_preds, fuse_probs = get_predictions(
        fusion, fuse_loader, graph_x, edge_index, has_node_ids=True)

    # ── Figure 1: Full results dashboard ──────────────────────────────────
    print("Plotting dashboard...")
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: signal samples
    ax_sigs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    plot_signal_samples(ax_sigs, fuse_test)
    fig.text(0.01, 0.97, "Sample signals per class (normalized, 1024 samples @ 12kHz)",
             color=MUTED, fontsize=10, va="top")

    # Row 1: confusion matrices
    ax_cm1 = fig.add_subplot(gs[1, :2])
    ax_cm2 = fig.add_subplot(gs[1, 2:])
    plot_confusion_matrix(ax_cm1, base_labels, base_preds, "Confusion matrix — CNN only")
    plot_confusion_matrix(ax_cm2, fuse_labels, fuse_preds, "Confusion matrix — CNN + GNN")

    # Row 2: ROC curves + ablation bar
    ax_roc1 = fig.add_subplot(gs[2, :2])
    ax_roc2 = fig.add_subplot(gs[2, 2:3])
    ax_abl  = fig.add_subplot(gs[2, 3])
    plot_roc_curves(ax_roc1, fuse_labels, fuse_probs, "ROC curves — CNN + GNN fusion")
    plot_roc_curves(ax_roc2, base_labels, base_probs, "ROC curves — CNN only")
    plot_ablation_bar(ax_abl)

    # Row 3: per-class metrics + confidence distributions
    ax_met1 = fig.add_subplot(gs[3, :2])
    ax_met2 = fig.add_subplot(gs[3, 2:])
    plot_per_class_metrics(ax_met1, fuse_labels, fuse_preds,
                           "Per-class metrics — CNN + GNN fusion")
    plot_confidence_distribution(ax_met2, fuse_labels, fuse_probs,
                                 "Confidence distribution — CNN + GNN fusion")

    # Title
    fig.suptitle("Industrial GNN Predictive Maintenance — Results Dashboard",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.995)

    path1 = "experiments/plots/results_dashboard.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"Saved: {path1}")
    plt.close()

    # ── Figure 2: Clean ablation comparison ───────────────────────────────
    print("Plotting ablation comparison...")
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.patch.set_facecolor(DARK_BG)
    fig2.suptitle("Ablation Study — CNN only vs CNN + GNN",
                  fontsize=15, fontweight="bold", color=TEXT_COLOR)

    plot_confusion_matrix(axes[0,0], base_labels, base_preds, "CNN only — confusion matrix")
    plot_confusion_matrix(axes[0,1], fuse_labels, fuse_preds, "CNN + GNN — confusion matrix")
    plot_ablation_bar(axes[0,2])
    plot_roc_curves(axes[1,0], base_labels, base_probs, "CNN only — ROC")
    plot_roc_curves(axes[1,1], fuse_labels, fuse_probs, "CNN + GNN — ROC")
    plot_per_class_metrics(axes[1,2], fuse_labels, fuse_preds, "CNN + GNN — per-class F1")

    plt.tight_layout()
    path2 = "experiments/plots/ablation_comparison.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"Saved: {path2}")
    plt.close()

    # ── Print summary ──────────────────────────────────────────────────────
    base_acc = (base_labels == base_preds).mean() * 100
    fuse_acc = (fuse_labels == fuse_preds).mean() * 100
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"  CNN only:    {base_acc:.2f}%")
    print(f"  CNN + GNN:   {fuse_acc:.2f}%")
    print(f"  Delta:       {fuse_acc - base_acc:+.2f}%")
    print(f"{'='*50}")
    print(f"\nPlots saved to experiments/plots/")


if __name__ == "__main__":
    main()
