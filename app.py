import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import scipy.io as sio
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.models.fusion_model import FusionModel
from src.data.graph_builder import build_graph, NODES, FAULT_TYPE_MAP

# ── Constants ─────────────────────────────────────────────────────────────
LABEL_NAMES  = {0: "Normal", 1: "Inner Race Fault", 2: "Ball Fault", 3: "Outer Race Fault"}
LABEL_COLORS = {0: "#1D9E75", 1: "#D85A30", 2: "#7F77DD", 3: "#EF9F27"}
WINDOW_SIZE  = 1024
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────
def load_model():
    model = FusionModel(num_classes=4).to(DEVICE)
    ckpt  = "experiments/best_fusion.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Loaded model from {ckpt}")
    else:
        print("WARNING: no checkpoint found — using random weights")
    model.eval()
    return model

graph      = build_graph()
graph_x    = graph.x.to(DEVICE)
edge_index = graph.edge_index.to(DEVICE)
model      = load_model()

NODE_LOOKUP = {
    (FAULT_TYPE_MAP[ft], sev, load): nid
    for nid, ft, sev, load, rpm in NODES
}


# ── Core prediction ───────────────────────────────────────────────────────
def predict_signal(signal: np.ndarray, node_id: int = 0):
    """Run inference on a 1D numpy signal array."""
    # Window + normalize
    signal = signal.flatten().astype(np.float32)
    if len(signal) < WINDOW_SIZE:
        signal = np.pad(signal, (0, WINDOW_SIZE - len(signal)))
    signal = signal[:WINDOW_SIZE]
    mean, std = signal.mean(), signal.std() + 1e-8
    signal = (signal - mean) / std

    x        = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0).to(DEVICE)
    node_ids = torch.tensor([node_id], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(x, node_ids, graph_x, edge_index)
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    pred_idx  = int(probs.argmax())
    pred_name = LABEL_NAMES[pred_idx]
    return probs, pred_idx, pred_name, signal


def plot_signal(signal: np.ndarray, pred_name: str, pred_idx: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.patch.set_facecolor('#0F0F13')
    color = LABEL_COLORS[pred_idx]

    # Signal plot
    ax1 = axes[0]
    ax1.set_facecolor('#1A1A20')
    ax1.plot(signal, color=color, linewidth=0.8, alpha=0.9)
    ax1.set_title("Vibration signal (normalized)", color='#D3D1C7', fontsize=11)
    ax1.set_xlabel("Sample", color='#888780')
    ax1.set_ylabel("Amplitude", color='#888780')
    ax1.tick_params(colors='#888780')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333330')

    # Confidence bar chart
    ax2 = axes[1]
    ax2.set_facecolor('#1A1A20')
    bars = ax2.barh(
        list(LABEL_NAMES.values()),
        [0,0,0,0],  # placeholder
        color=[LABEL_COLORS[i] for i in range(4)],
        alpha=0.3,
        height=0.5,
    )
    # Actual bars
    probs_plot, _, _, _ = predict_signal(signal * signal.std() + signal.mean())
    for bar, prob in zip(bars, probs_plot):
        bar.set_width(prob)
        bar.set_alpha(0.85)

    ax2.set_xlim(0, 1)
    ax2.set_title("Confidence scores", color='#D3D1C7', fontsize=11)
    ax2.set_xlabel("Probability", color='#888780')
    ax2.tick_params(colors='#D3D1C7')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333330')

    plt.tight_layout()
    return fig


# ── Gradio handlers ───────────────────────────────────────────────────────
def predict_from_mat(file_obj):
    if file_obj is None:
        return "No file uploaded.", None

    try:
        mat  = sio.loadmat(file_obj.name)
        keys = [k for k in mat.keys() if not k.startswith('_') and 'DE_time' in k]
        if not keys:
            keys = [k for k in mat.keys() if not k.startswith('_')]
        signal = mat[keys[0]].flatten()
    except Exception as e:
        return f"Error reading file: {e}", None

    probs, pred_idx, pred_name, norm_signal = predict_signal(signal)

    result = f"## Prediction: {pred_name}\n\n"
    result += "| Fault type | Confidence |\n|---|---|\n"
    for i, (name, prob) in enumerate(zip(LABEL_NAMES.values(), probs)):
        marker = " ◀" if i == pred_idx else ""
        result += f"| {name} | {prob*100:.1f}%{marker} |\n"

    fig = plot_signal(norm_signal, pred_name, pred_idx)
    return result, fig


def predict_from_random(fault_type: str):
    """Generate a synthetic signal for demo purposes."""
    np.random.seed(42)
    t      = np.linspace(0, 1, WINDOW_SIZE)
    signal = np.random.randn(WINDOW_SIZE) * 0.3  # noise

    fault_map = {
        "Normal":            (0,   []),
        "Inner Race Fault":  (1,   [97.0]),
        "Ball Fault":        (2,   [141.0]),
        "Outer Race Fault":  (3,   [63.0]),
    }
    label_idx, freqs = fault_map[fault_type]

    # Add characteristic fault frequency impulses
    for freq in freqs:
        signal += 0.8 * np.sin(2 * np.pi * freq * t)
        # Add impulses
        impulse_times = np.arange(0, WINDOW_SIZE, int(12000/freq))
        for imp in impulse_times:
            if imp < WINDOW_SIZE:
                signal[imp] += np.random.choice([-1, 1]) * 1.5

    node_id = NODE_LOOKUP.get((label_idx, 7 if label_idx > 0 else 0, 0), 0)
    probs, pred_idx, pred_name, norm_signal = predict_signal(signal, node_id)

    result = f"## Prediction: {pred_name}\n\n"
    result += f"*(Ground truth: {fault_type})*\n\n"
    result += "| Fault type | Confidence |\n|---|---|\n"
    for i, (name, prob) in enumerate(zip(LABEL_NAMES.values(), probs)):
        marker = " ◀" if i == pred_idx else ""
        result += f"| {name} | {prob*100:.1f}%{marker} |\n"

    fig = plot_signal(norm_signal, pred_name, pred_idx)
    return result, fig


# ── UI ────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Bearing Fault Detection") as demo:
    gr.Markdown("""
    # Bearing Fault Detection — CNN + GNN Fusion
    **Industrial predictive maintenance** · CWRU Benchmark · RTX 3060
    Upload a CWRU `.mat` file or generate a synthetic signal to predict bearing fault type.
    """)

    with gr.Tabs():
        # Tab 1: Upload .mat file
        with gr.TabItem("Upload .mat file"):
            with gr.Row():
                file_input = gr.File(label="Upload CWRU .mat file", file_types=[".mat"])
            with gr.Row():
                mat_btn = gr.Button("Run Inference", variant="primary")
            with gr.Row():
                mat_result = gr.Markdown()
                mat_plot   = gr.Plot()
            mat_btn.click(predict_from_mat, inputs=file_input, outputs=[mat_result, mat_plot])

        # Tab 2: Synthetic signal demo
        with gr.TabItem("Synthetic signal demo"):
            gr.Markdown("Generate a synthetic vibration signal with known fault characteristics.")
            with gr.Row():
                fault_selector = gr.Radio(
                    choices=["Normal", "Inner Race Fault", "Ball Fault", "Outer Race Fault"],
                    value="Normal",
                    label="Select fault type to simulate",
                )
            with gr.Row():
                demo_btn = gr.Button("Generate & Predict", variant="primary")
            with gr.Row():
                demo_result = gr.Markdown()
                demo_plot   = gr.Plot()
            demo_btn.click(predict_from_random, inputs=fault_selector, outputs=[demo_result, demo_plot])

    gr.Markdown("""
    ---
    **Model:** CNN + GNN Fusion · **Test accuracy:** 99.10% · **Ablation delta:** +0.61% vs CNN-only
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Base())
