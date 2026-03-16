import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.io as sio
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.fusion_model import FusionModel
from src.data.graph_builder import build_graph, NODES, FAULT_TYPE_MAP
from src.data.cwru_dataset_v2 import LABEL_NAMES

LABEL_LIST  = list(LABEL_NAMES.values())
COLORS      = ["#1D9E75", "#7F77DD", "#D85A30", "#EF9F27"]
DARK        = "#0F0F13"; PANEL = "#1A1A22"; WHITE = "#F0EEE8"; MUTED = "#888780"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW      = 1024
STRIDE      = 512

NODE_LOOKUP = {
    (FAULT_TYPE_MAP[ft], sev, load): nid
    for nid, ft, sev, load, rpm in NODES
}

# ── Load model ─────────────────────────────────────────────────────────────
graph      = build_graph()
graph_x    = graph.x.to(DEVICE)
edge_index = graph.edge_index.to(DEVICE)

model = FusionModel(num_classes=4).to(DEVICE)
if os.path.exists("experiments/best_fusion.pt"):
    model.load_state_dict(torch.load("experiments/best_fusion.pt",
                                      map_location=DEVICE))
model.eval()
print(f"Model loaded on {DEVICE}")


# ── Core inference ──────────────────────────────────────────────────────────
def predict_window(window, node_id=0):
    w         = window.astype(np.float32)
    mean, std = w.mean(), w.std() + 1e-8
    norm      = (w - mean) / std
    x         = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    nids      = torch.tensor([node_id], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(x, nids, graph_x, edge_index)
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs, norm


def get_signal(file_obj):
    mat  = sio.loadmat(file_obj.name)
    for k in mat.keys():
        if k.startswith('_'): continue
        arr = mat[k]
        if arr.size < WINDOW: continue
        sig = arr.flatten().astype(np.float32)
        if len(sig) >= WINDOW:
            return sig
    return None


def majority_vote(signal, node_id=0, n=20):
    """Run inference on n evenly spaced windows, return averaged probs."""
    step      = max(STRIDE, len(signal) // n)
    all_probs = []
    for i in range(n):
        start = i * step
        if start + WINDOW > len(signal): break
        _, p, _ = predict_window(signal[start:start+WINDOW], node_id)
        all_probs.append(p)
    probs = np.mean(all_probs, axis=0)
    return int(probs.argmax()), probs


# ── Tab 1: Single sample prediction ────────────────────────────────────────
def predict_single(file_obj):
    if file_obj is None:
        return "No file uploaded.", None

    signal = get_signal(file_obj)
    if signal is None:
        return "Could not read signal.", None

    pred, probs = majority_vote(signal, node_id=0, n=20)
    _, _, norm  = predict_window(signal[:WINDOW], node_id=0)

    pred_name  = LABEL_NAMES[pred]
    confidence = probs[pred] * 100
    is_fault   = pred > 0
    verdict    = "⚠  FAULT DETECTED" if is_fault else "✓  NORMAL — NO FAULT"
    vcolor     = "#D85A30" if is_fault else "#1D9E75"

    result_md = f"## {verdict}\n\n"
    result_md += f"**Predicted class:** {pred_name}  \n"
    result_md += f"**Confidence:** {confidence:.1f}%  \n\n"
    result_md += "| Fault type | Confidence |\n|---|---|\n"
    for i, (name, p) in enumerate(zip(LABEL_LIST, probs)):
        marker = " ◀" if i == pred else ""
        result_md += f"| {name} | {p*100:.1f}%{marker} |\n"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor(DARK)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor("#2A2A32")
        ax.tick_params(colors=MUTED)

    axes[0].plot(signal[:WINDOW], color=COLORS[pred], lw=1.0)
    axes[0].set_title("Raw signal window", color=WHITE, fontsize=11)
    axes[0].set_xlabel("Sample", color=MUTED)
    axes[0].set_ylabel("Amplitude", color=MUTED)
    axes[0].set_xlim(0, WINDOW); axes[0].grid(True, alpha=0.3)

    axes[1].plot(norm, color=COLORS[pred], lw=1.0)
    axes[1].set_title("Normalized input", color=WHITE, fontsize=11)
    axes[1].set_xlabel("Sample", color=MUTED)
    axes[1].set_xlim(0, WINDOW); axes[1].grid(True, alpha=0.3)
    axes[1].text(0.5, 0.05, verdict,
                 transform=axes[1].transAxes, ha="center",
                 color=vcolor, fontsize=12, fontweight="bold",
                 bbox=dict(fc=PANEL, ec=vcolor, lw=2,
                           boxstyle="round,pad=0.4"))

    bar_colors = [COLORS[i] if i==pred else "#2A2A32" for i in range(4)]
    bars = axes[2].bar(LABEL_LIST, probs*100, color=bar_colors,
                       alpha=0.9, edgecolor=DARK)
    bars[pred].set_edgecolor(WHITE); bars[pred].set_linewidth(2.5)
    for bar, p in zip(bars, probs):
        axes[2].text(bar.get_x()+bar.get_width()/2, p*100+0.5,
                     f"{p*100:.1f}%", ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color=WHITE)
    axes[2].set_ylim(0, 115)
    axes[2].set_title("Confidence scores", color=WHITE, fontsize=11)
    axes[2].set_ylabel("Confidence (%)", color=MUTED)
    axes[2].grid(True, axis="y", alpha=0.3)
    for sp in axes[2].spines.values(): sp.set_edgecolor("#2A2A32")
    axes[2].tick_params(colors=MUTED)

    plt.tight_layout()
    return result_md, fig


# ── Tab 2: Signal visualizer ────────────────────────────────────────────────
def visualize_signal(file_obj):
    if file_obj is None:
        return None

    signal = get_signal(file_obj)
    if signal is None:
        return None

    offsets = [i * (len(signal)//4) for i in range(4)]
    offsets = [o for o in offsets if o + WINDOW <= len(signal)][:4]

    fig, axes = plt.subplots(len(offsets), 3, figsize=(16, 4*len(offsets)))
    if len(offsets) == 1: axes = axes.reshape(1, -1)
    fig.patch.set_facecolor(DARK)
    fig.suptitle("Signal Analysis — 4 windows across the file",
                 color=WHITE, fontsize=13, fontweight="bold")

    for row, offset in enumerate(offsets):
        window = signal[offset:offset+WINDOW]
        pred, probs, norm = predict_window(window)
        color    = COLORS[pred]
        is_fault = pred > 0

        for ax in axes[row]:
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor("#2A2A32")
            ax.tick_params(colors=MUTED)

        axes[row,0].plot(window, color=color, lw=0.8)
        axes[row,0].set_title(f"Window {row+1} — raw", color=WHITE, fontsize=10)
        axes[row,0].set_xlim(0, WINDOW); axes[row,0].grid(True, alpha=0.3)
        axes[row,0].set_ylabel("Amplitude", color=MUTED)

        verdict = "⚠ FAULT" if is_fault else "✓ NORMAL"
        vcolor  = "#D85A30" if is_fault else "#1D9E75"
        axes[row,1].plot(norm, color=color, lw=0.8)
        axes[row,1].set_title(f"Normalized — {LABEL_NAMES[pred]}",
                              color=WHITE, fontsize=10)
        axes[row,1].set_xlim(0, WINDOW); axes[row,1].grid(True, alpha=0.3)
        axes[row,1].text(0.02, 0.95, verdict,
                         transform=axes[row,1].transAxes,
                         color=vcolor, fontsize=11, fontweight="bold",
                         va="top",
                         bbox=dict(fc=PANEL, ec=vcolor, lw=1.5,
                                   boxstyle="round,pad=0.3"))

        freqs = np.fft.rfftfreq(WINDOW, d=1/12000)
        fft   = np.abs(np.fft.rfft(norm))
        axes[row,2].plot(freqs[:400], fft[:400], color=color, lw=0.8)
        axes[row,2].set_title("Frequency spectrum", color=WHITE, fontsize=10)
        axes[row,2].set_xlabel("Hz", color=MUTED)
        axes[row,2].set_xlim(0, freqs[400]); axes[row,2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Tab 3: Synthetic demo ───────────────────────────────────────────────────
def predict_synthetic(fault_type):
    np.random.seed(42)
    t      = np.linspace(0, 1, WINDOW)
    signal = np.random.randn(WINDOW).astype(np.float32) * 0.3

    fault_map = {
        "Normal":           (0, []),
        "Inner Race Fault": (1, [97.0]),
        "Ball Fault":       (2, [141.0]),
        "Outer Race Fault": (3, [63.0]),
    }
    true_label, freqs = fault_map[fault_type]
    for freq in freqs:
        signal += 0.8 * np.sin(2*np.pi*freq*t).astype(np.float32)
        for imp in np.arange(0, WINDOW, int(12000/freq)):
            if imp < WINDOW:
                signal[int(imp)] += np.random.choice([-1,1]) * 1.5

    node_id          = NODE_LOOKUP.get((true_label, 7, 0), 0)
    pred, probs, norm= predict_window(signal, node_id)
    pred_name        = LABEL_NAMES[pred]
    correct          = pred == true_label
    verdict          = "✓ CORRECT" if correct else "✗ WRONG"
    vcolor           = "#1D9E75" if correct else "#D85A30"
    is_fault         = pred > 0
    diag             = "⚠ FAULT DETECTED" if is_fault else "✓ NO FAULT"

    result_md  = f"## {verdict} — Ground truth: {fault_type}\n\n"
    result_md += f"**Predicted:** {pred_name} ({probs[pred]*100:.1f}%)\n\n"
    result_md += f"**Diagnosis:** {diag}\n\n"
    result_md += "| Fault type | Confidence |\n|---|---|\n"
    for i, (name, p) in enumerate(zip(LABEL_LIST, probs)):
        marker = " ◀" if i == pred else ""
        result_md += f"| {name} | {p*100:.1f}%{marker} |\n"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor(DARK)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor("#2A2A32")
        ax.tick_params(colors=MUTED)

    axes[0].plot(norm, color=COLORS[true_label], lw=0.9)
    axes[0].set_title(f"Synthetic signal — {fault_type}", color=WHITE, fontsize=11)
    axes[0].set_xlim(0, WINDOW); axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.95, verdict,
                 transform=axes[0].transAxes, va="top",
                 color=vcolor, fontsize=12, fontweight="bold",
                 bbox=dict(fc=PANEL, ec=vcolor, lw=2, boxstyle="round,pad=0.4"))

    bar_colors = [COLORS[i] if i==pred else "#2A2A32" for i in range(4)]
    bars = axes[1].bar(LABEL_LIST, probs*100, color=bar_colors,
                       alpha=0.9, edgecolor=DARK)
    bars[pred].set_edgecolor(WHITE); bars[pred].set_linewidth(2.5)
    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_x()+bar.get_width()/2, p*100+0.5,
                     f"{p*100:.1f}%", ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color=WHITE)
    axes[1].set_ylim(0, 115)
    axes[1].set_title("Confidence scores", color=WHITE, fontsize=11)
    axes[1].set_ylabel("Confidence (%)", color=MUTED)
    axes[1].grid(True, axis="y", alpha=0.3)
    for sp in axes[1].spines.values(): sp.set_edgecolor("#2A2A32")
    axes[1].tick_params(colors=MUTED)

    plt.tight_layout()
    return result_md, fig


# ── Tab 4: Results dashboard ────────────────────────────────────────────────
def load_plot(name):
    path = f"experiments/plots/{name}"
    return path if os.path.exists(path) else None


# ── UI ──────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Bearing Fault Detection") as demo:
    gr.Markdown("""
    # Bearing Fault Detection — CNN + GNN Fusion
    **Industrial predictive maintenance** · CWRU Benchmark · RTX 3060 · 99.10% accuracy
    """)

    with gr.Tabs():

        with gr.TabItem("Single sample prediction"):
            gr.Markdown("""
            Upload a CWRU `.mat` file — get a fault diagnosis verdict with confidence scores.
            Uses majority voting over 20 windows for robust prediction.
            """)
            file_input  = gr.File(label="Upload .mat file", file_types=[".mat"])
            predict_btn = gr.Button("Run Diagnosis", variant="primary")
            with gr.Row():
                result_md   = gr.Markdown()
                result_plot = gr.Plot()
            predict_btn.click(predict_single,
                              inputs=file_input,
                              outputs=[result_md, result_plot])

        with gr.TabItem("Signal visualizer"):
            gr.Markdown("""
            Upload a `.mat` file to see raw signal, normalized input,
            frequency spectrum and prediction across 4 windows.
            """)
            vis_file = gr.File(label="Upload .mat file", file_types=[".mat"])
            vis_btn  = gr.Button("Visualize", variant="primary")
            vis_plot = gr.Plot()
            vis_btn.click(visualize_signal, inputs=vis_file, outputs=vis_plot)

        with gr.TabItem("Synthetic signal demo"):
            gr.Markdown("Generate a synthetic fault signal and test the model.")
            fault_sel = gr.Radio(
                ["Normal", "Inner Race Fault", "Ball Fault", "Outer Race Fault"],
                value="Normal", label="Fault type to simulate")
            syn_btn  = gr.Button("Generate & Predict", variant="primary")
            with gr.Row():
                syn_md   = gr.Markdown()
                syn_plot = gr.Plot()
            syn_btn.click(predict_synthetic,
                          inputs=fault_sel,
                          outputs=[syn_md, syn_plot])

        with gr.TabItem("Results dashboard"):
            gr.Markdown("""
            ## Experiment results
            Run `python src/eval/full_evaluation.py` to regenerate plots.
            """)
            gr.Image(load_plot("fig_three_numbers.png"),
                     label="Ablation study — CNN only vs CNN+GNN")
            gr.Image(load_plot("fig_cwru_dashboard.png"),
                     label="CWRU test set — confusion matrix + ROC + per-class F1")
            gr.Image(load_plot("fig_signal_visualizer.png"),
                     label="Signal visualizer — one sample per fault class")
            gr.Image(load_plot("cross_dataset_susu.png"),
                     label="Cross-dataset test — domain shift analysis")

    gr.Markdown("""
    ---
    **Test accuracy:** 99.10% (CNN+GNN) vs 98.49% (CNN only) · **Dataset:** CWRU 20 files · 6,622 windows
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
