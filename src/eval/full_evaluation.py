"""
Full evaluation pipeline — generates all result plots:
  Fig 1: Three-number story (CWRU / zero-shot / fine-tuned)
  Fig 2: CWRU test dashboard (confusion + ROC + per-class)
  Fig 3: SuSu fine-tuned dashboard (confusion + ROC + per-class)
  Fig 4: Signal visualizer (raw + normalized + FFT + prediction)
  Fig 5: Single sample prediction card
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from src.models.fusion_model import FusionModel
from src.data.cwru_dataset_v2 import CWRUDatasetV2, LABEL_NAMES
from src.data.graph_builder import build_graph, NODES, FAULT_TYPE_MAP
from src.train.finetune_susu import SuSuDataset

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DARK    = "#0F0F13"; WHITE = "#F0EEE8"; MUTED = "#888780"; PANEL = "#1A1A22"
TEAL    = "#1D9E75"; PUR   = "#7F77DD"; COR   = "#D85A30"; AMB   = "#EF9F27"
COLORS  = [TEAL, PUR, COR, AMB]
LNAMES  = list(LABEL_NAMES.values())
GRID    = "#2A2A32"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": WHITE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": WHITE, "grid.color": GRID, "grid.alpha": 0.4,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID,
})

NODE_LOOKUP = {
    (FAULT_TYPE_MAP[ft], sev, load): nid
    for nid, ft, sev, load, rpm in NODES
}


# ── Inference helpers ──────────────────────────────────────────────────────
@torch.no_grad()
def infer_loader(model, loader, graph_x, edge_index, has_node_ids=True):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        if has_node_ids:
            x, y, nids = batch
            x, y, nids = x.to(DEVICE), y.to(DEVICE), nids.to(DEVICE)
            logits = model(x, nids, graph_x, edge_index)
        else:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


@torch.no_grad()
def infer_single(model, signal_window, node_id, graph_x, edge_index):
    """Run inference on a single 1024-sample window."""
    model.eval()
    w    = signal_window.copy().astype(np.float32)
    mean, std = w.mean(), w.std() + 1e-8
    w    = (w - mean) / std
    x    = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).to(DEVICE)
    nids = torch.tensor([node_id], dtype=torch.long).to(DEVICE)
    logits = model(x, nids, graph_x, edge_index)
    probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred   = int(probs.argmax())
    return pred, probs, w


# ── Plot helpers ───────────────────────────────────────────────────────────
def plot_confusion(ax, labels, preds, title):
    cm  = confusion_matrix(labels, preds)
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["Norm","IR","Ball","OR"], fontsize=9)
    ax.set_yticklabels(["Norm","IR","Ball","OR"], fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, pad=8, fontsize=11)
    for i in range(4):
        for j in range(4):
            v = cmn[i, j]
            ax.text(j, i, f"{v:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else WHITE, fontweight="bold")


def plot_roc(ax, labels, probs, title):
    lb = label_binarize(labels, classes=[0,1,2,3])
    for i, (name, color) in enumerate(zip(LNAMES, COLORS)):
        fpr, tpr, _ = roc_curve(lb[:,i], probs[:,i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} ({roc_auc:.3f})")
    ax.plot([0,1],[0,1], color=MUTED, lw=1, ls="--", alpha=0.5)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title, pad=8, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True)


def plot_perclass(ax, labels, preds, title):
    report = classification_report(labels, preds,
                                   target_names=LNAMES, output_dict=True)
    x      = np.arange(4)
    w      = 0.25
    metrics= ["precision", "recall", "f1-score"]
    mcolors= [TEAL, PUR, AMB]
    for i, (metric, color) in enumerate(zip(metrics, mcolors)):
        vals = [report[cls][metric] for cls in LNAMES]
        bars = ax.bar(x + i*w, vals, w, label=metric.capitalize(),
                      color=color, alpha=0.85, edgecolor=DARK, lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=WHITE)
    ax.set_xticks(x+w); ax.set_xticklabels(LNAMES, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title(title, pad=8, fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, axis="y")


def plot_signal_panel(ax_raw, ax_norm, ax_fft, signal, norm_signal,
                      label_name, pred_name, probs, color):
    """Three-panel signal view: raw / normalized / FFT."""
    # Raw
    ax_raw.plot(signal[:1024], color=color, lw=0.8, alpha=0.9)
    ax_raw.set_title(f"Raw signal — {label_name}", fontsize=10, color=color)
    ax_raw.set_xlabel("Sample"); ax_raw.set_ylabel("Amplitude")
    ax_raw.set_xlim(0, 1024); ax_raw.grid(True)

    # Normalized
    ax_norm.plot(norm_signal, color=color, lw=0.8, alpha=0.9)
    ax_norm.set_title(f"Normalized (model input)", fontsize=10)
    ax_norm.set_xlabel("Sample"); ax_norm.set_ylabel("Amplitude")
    ax_norm.set_xlim(0, 1024); ax_norm.grid(True)
    ax_norm.text(0.97, 0.95,
                 f"Pred: {pred_name}",
                 transform=ax_norm.transAxes, ha="right", va="top",
                 color=TEAL if pred_name==label_name else COR,
                 fontsize=10, fontweight="bold",
                 bbox=dict(fc=PANEL, ec=GRID, lw=0.8, boxstyle="round,pad=0.3"))

    # FFT
    freqs = np.fft.rfftfreq(1024, d=1/12000)
    fft   = np.abs(np.fft.rfft(norm_signal))
    ax_fft.plot(freqs[:400], fft[:400], color=color, lw=0.8, alpha=0.9)
    ax_fft.set_title("Frequency spectrum (FFT)", fontsize=10)
    ax_fft.set_xlabel("Frequency (Hz)"); ax_fft.set_ylabel("Magnitude")
    ax_fft.set_xlim(0, freqs[400]); ax_fft.grid(True)

    # Confidence bars overlay on FFT
    ax_conf = ax_fft.twinx()
    bars = ax_conf.bar(np.arange(4)*0.15 + 0.6,
                       probs, 0.12,
                       color=COLORS, alpha=0.7, transform=ax_fft.transAxes)
    for bar, p, name in zip(bars, probs, LNAMES):
        ax_conf.text(bar.get_x() + bar.get_width()/2,
                     p + 0.02, f"{p*100:.0f}%",
                     ha="center", va="bottom", fontsize=7,
                     color=WHITE, transform=ax_fft.transAxes)
    ax_conf.set_ylim(0, 1.4)
    ax_conf.set_yticks([])


# ── FIG 1: Three-number story ─────────────────────────────────────────────
def fig_three_numbers():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK)

    stages  = ["CWRU\nin-domain\n(CNN+GNN)",
               "SuSu\nzero-shot\n(no adapt)",
               "SuSu\nfine-tuned\n(20% data)"]
    accs    = [99.10, 24.80, 92.31]
    colors  = [TEAL, COR, AMB]
    bars    = ax.bar(stages, accs, color=colors, alpha=0.88,
                     edgecolor=DARK, width=0.4, linewidth=0.5)

    for bar, acc, color in zip(bars, accs, colors):
        ax.text(bar.get_x()+bar.get_width()/2, acc+0.5,
                f"{acc:.2f}%", ha="center", va="bottom",
                fontsize=16, fontweight="bold", color=WHITE)

    # Annotations
    ax.annotate("", xy=(1, 24.80), xytext=(0, 99.10),
                arrowprops=dict(arrowstyle="->", color=COR, lw=2.0))
    ax.text(0.5, 65, "Domain shift\n−74.3pp", ha="center",
            color=COR, fontsize=10, fontweight="bold")

    ax.annotate("", xy=(2, 92.31), xytext=(1, 24.80),
                arrowprops=dict(arrowstyle="->", color=AMB, lw=2.0))
    ax.text(1.5, 62, "Fine-tuning\n+67.5pp", ha="center",
            color=AMB, fontsize=10, fontweight="bold")

    ax.axhline(25, color=MUTED, lw=1, ls=":", alpha=0.6)
    ax.text(2.3, 26.5, "random (4-class)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Complete Transfer Learning Story\n"
                 "Train on CWRU → Deploy on new machine → Adapt with 20% data",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = "experiments/plots/fig_three_numbers.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  Saved: {path}")


# ── FIG 2 & 3: Full dashboards ────────────────────────────────────────────
def fig_dashboard(labels, preds, probs, acc, title, fname):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f"{title} — Test Accuracy: {acc*100:.2f}%",
                 fontsize=14, fontweight="bold", color=WHITE)

    plot_confusion(fig.add_subplot(gs[0,:2]),  labels, preds,
                   "Confusion matrix (normalised)")
    plot_roc(      fig.add_subplot(gs[0, 2]),  labels, probs, "ROC curves")
    plot_perclass( fig.add_subplot(gs[1, :]),  labels, preds,
                   "Per-class precision / recall / F1")

    path = f"experiments/plots/{fname}"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  Saved: {path}")


# ── FIG 4: Signal visualizer ──────────────────────────────────────────────
def fig_signal_visualizer(model, graph_x, edge_index):
    """Show one sample per class from CWRU test set with full signal panel."""
    test_ds = CWRUDatasetV2(split="test")

    # Collect one sample per class
    samples = {}
    for x, y in test_ds:
        label = y.item()
        if label not in samples:
            samples[label] = x.squeeze().numpy()
        if len(samples) == 4:
            break

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK)
    fig.suptitle("Signal Visualizer — One sample per fault class (CWRU test set)",
                 fontsize=13, fontweight="bold", color=WHITE)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    for row, (label, norm_sig) in enumerate(sorted(samples.items())):
        node_id   = NODE_LOOKUP.get((label, 7, 0), 0)
        pred, probs, _ = infer_single(model, norm_sig, node_id,
                                      graph_x, edge_index)
        label_name = LABEL_NAMES[label]
        pred_name  = LABEL_NAMES[pred]
        color      = COLORS[label]
        correct    = pred == label

        # Raw (denormalized approximation — just show norm for simplicity)
        ax_raw  = fig.add_subplot(gs[row, 0])
        ax_norm = fig.add_subplot(gs[row, 1])
        ax_fft  = fig.add_subplot(gs[row, 2])

        # Raw signal panel
        ax_raw.plot(norm_sig, color=color, lw=0.8, alpha=0.9)
        verdict = "FAULT DETECTED" if label > 0 else "NORMAL"
        verdict_color = COR if label > 0 else TEAL
        ax_raw.set_title(f"Class: {label_name}", fontsize=10,
                         color=color, fontweight="bold")
        ax_raw.set_xlim(0, 1024); ax_raw.grid(True)
        ax_raw.set_ylabel("Amplitude")

        # Prediction panel
        correct_str = "✓ CORRECT" if correct else "✗ WRONG"
        correct_col = TEAL if correct else COR
        ax_norm.plot(norm_sig, color=color, lw=0.8, alpha=0.9)
        ax_norm.set_title(f"Prediction: {pred_name}", fontsize=10)
        ax_norm.set_xlim(0, 1024); ax_norm.grid(True)
        ax_norm.text(0.02, 0.95, correct_str,
                     transform=ax_norm.transAxes,
                     color=correct_col, fontsize=11, fontweight="bold",
                     va="top",
                     bbox=dict(fc=PANEL, ec=correct_col,
                               lw=1.5, boxstyle="round,pad=0.3"))

        # Confidence bars
        ax_fft.bar(LNAMES, probs, color=COLORS, alpha=0.85,
                   edgecolor=DARK, lw=0.5)
        for i, (p, name) in enumerate(zip(probs, LNAMES)):
            ax_fft.text(i, p+0.01, f"{p*100:.1f}%",
                        ha="center", va="bottom", fontsize=9,
                        color=WHITE, fontweight="bold")
        ax_fft.set_ylim(0, 1.15)
        ax_fft.set_title("Model confidence", fontsize=10)
        ax_fft.set_ylabel("Probability"); ax_fft.grid(True, axis="y")
        # Highlight predicted bar
        ax_fft.patches[pred].set_edgecolor(WHITE)
        ax_fft.patches[pred].set_linewidth(2.5)

    path = "experiments/plots/fig_signal_visualizer.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  Saved: {path}")


# ── FIG 5: Single sample card ─────────────────────────────────────────────
def fig_single_prediction(model, graph_x, edge_index,
                          mat_path, true_label, title):
    """Predict one real .mat file window and render a verdict card."""
    mat    = sio.loadmat(mat_path)
    key    = [k for k in mat.keys() if not k.startswith('_')][0]
    signal = mat[key].flatten() if mat[key].ndim == 1 \
             else mat[key][0].flatten()
    mid = max(0, len(signal)//2 - 512)
    window = signal[mid:mid+1024].astype(np.float32)
    if len(window) < 1024: window = np.pad(window, (0, 1024-len(window)))

    node_id = NODE_LOOKUP.get((true_label, 7, 0), 0)
    pred, probs, norm_sig = infer_single(model, window, node_id,
                                         graph_x, edge_index)

    label_name = LABEL_NAMES[true_label]
    pred_name  = LABEL_NAMES[pred]
    correct    = pred == true_label

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK)

    verdict     = "✓  CORRECT DIAGNOSIS" if correct else "✗  WRONG DIAGNOSIS"
    vcolor      = TEAL if correct else COR
    fault_str   = "FAULT DETECTED" if pred > 0 else "NO FAULT — NORMAL"
    fault_color = COR if pred > 0 else TEAL

    fig.suptitle(f"{title}\n{verdict}  —  True: {label_name}  →  "
                 f"Predicted: {pred_name}",
                 fontsize=13, fontweight="bold", color=vcolor)

    # Raw signal
    ax1 = axes[0]
    ax1.plot(window, color=COLORS[true_label], lw=1.0, alpha=0.9)
    ax1.set_title("Raw signal window (1024 samples)", fontsize=10)
    ax1.set_xlabel("Sample index"); ax1.set_ylabel("Amplitude (raw)")
    ax1.set_xlim(0, 1024); ax1.grid(True)

    # Normalized signal
    ax2 = axes[1]
    ax2.plot(norm_sig, color=COLORS[pred], lw=1.0, alpha=0.9)
    ax2.set_title("Normalized input (what the model sees)", fontsize=10)
    ax2.set_xlabel("Sample index"); ax2.set_ylabel("Amplitude (norm)")
    ax2.set_xlim(0, 1024); ax2.grid(True)
    ax2.text(0.5, 0.05, fault_str,
             transform=ax2.transAxes, ha="center", va="bottom",
             color=fault_color, fontsize=13, fontweight="bold",
             bbox=dict(fc=PANEL, ec=fault_color, lw=2,
                       boxstyle="round,pad=0.5"))

    # Confidence bars
    ax3 = axes[2]
    bar_colors = [TEAL if i==pred else "#2A2A32" for i in range(4)]
    bars = ax3.bar(LNAMES, probs*100, color=bar_colors,
                   alpha=0.9, edgecolor=DARK, lw=0.5)
    bars[pred].set_edgecolor(WHITE)
    bars[pred].set_linewidth(2.5)
    for bar, p in zip(bars, probs):
        ax3.text(bar.get_x()+bar.get_width()/2, p*100+0.5,
                 f"{p*100:.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=WHITE)
    ax3.set_ylim(0, 115)
    ax3.set_ylabel("Confidence (%)")
    ax3.set_title(f"Confidence scores\n→ {pred_name} ({probs[pred]*100:.1f}%)",
                  fontsize=10)
    ax3.grid(True, axis="y")

    plt.tight_layout()
    safe_title = title.replace(" ","_").replace("/","_")
    path = f"experiments/plots/fig_single_{safe_title}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs("experiments/plots", exist_ok=True)
    print(f"Device: {DEVICE}")

    # Load graph
    graph      = build_graph()
    graph_x    = graph.x.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)

    # Load models
    cwru_model = FusionModel(num_classes=4).to(DEVICE)
    cwru_model.load_state_dict(torch.load("experiments/best_fusion.pt",
                                          map_location=DEVICE))

    ft_model = FusionModel(num_classes=4).to(DEVICE)
    if os.path.exists("experiments/best_fusion_finetuned.pt"):
        ft_model.load_state_dict(torch.load(
            "experiments/best_fusion_finetuned.pt", map_location=DEVICE))
    else:
        print("WARNING: no fine-tuned model found, using CWRU model")
        ft_model = cwru_model

    print("\nGenerating all figures...")

    # Fig 1: three-number story
    print("\n[1/5] Three-number story")
    fig_three_numbers()

    # Fig 2: CWRU dashboard
    print("[2/5] CWRU test dashboard")
    test_ds    = CWRUDatasetV2(split="test")
    test_loader= DataLoader(test_ds, batch_size=128, shuffle=False,
                            num_workers=4)

    # Need fusion dataset for CWRU
    from src.train.train_fusion import CWRUFusionDataset
    fuse_test  = CWRUFusionDataset("test")
    fuse_loader= DataLoader(fuse_test, batch_size=128, shuffle=False,
                            num_workers=4)
    labels, preds, probs = infer_loader(cwru_model, fuse_loader,
                                        graph_x, edge_index, has_node_ids=True)
    fig_dashboard(labels, preds, probs, (preds==labels).mean(),
                  "CWRU Test Set — CNN+GNN Fusion",
                  "fig_cwru_dashboard.png")

    # Fig 3: SuSu fine-tuned dashboard
    print("[3/5] SuSu fine-tuned dashboard")
    susu_test  = SuSuDataset("test")
    susu_loader= DataLoader(susu_test, batch_size=128, shuffle=False,
                            num_workers=4)
    labels2, preds2, probs2 = infer_loader(ft_model, susu_loader,
                                           graph_x, edge_index,
                                           has_node_ids=True)
    fig_dashboard(labels2, preds2, probs2, (preds2==labels2).mean(),
                  "SuSu Test Set (18Hz) — Fine-tuned Model",
                  "fig_susu_dashboard.png")

    # Fig 4: Signal visualizer
    print("[4/5] Signal visualizer")
    fig_signal_visualizer(cwru_model, graph_x, edge_index)

    # Fig 5: Single sample predictions (one per fault type)
    print("[5/5] Single sample predictions")
    singles = [
        ("data/raw/CWRU/Normal/97_Normal_0.mat",                              0, "CWRU_Normal"),
        ("data/raw/CWRU/12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat",   1, "CWRU_InnerRace"),
        ("data/raw/CWRU/12k_Drive_End_Bearing_Fault_Data/B/007/118_0.mat",    2, "CWRU_Ball"),
        ("data/raw/CWRU/12k_Drive_End_Bearing_Fault_Data/OR/007/@6/130@6_0.mat", 3, "CWRU_OuterRace"),
    ]
    for mat_path, true_label, title in singles:
        if os.path.exists(mat_path):
            fig_single_prediction(cwru_model, graph_x, edge_index,
                                  mat_path, true_label, title)

    print("\n✓ All figures saved to experiments/plots/")
    print("Files:")
    for f in sorted(os.listdir("experiments/plots")):
        size = os.path.getsize(f"experiments/plots/{f}") // 1024
        print(f"  {f}  ({size} KB)")


if __name__ == "__main__":
    main()
