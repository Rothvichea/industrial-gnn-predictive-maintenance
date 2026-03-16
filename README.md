# Industrial GNN Predictive Maintenance

Bearing fault detection using **multimodal fusion** of vibration signal encoding (1D-CNN) and equipment graph context (GraphSAGE GNN) on the CWRU benchmark dataset.

> Developed with AI-assisted pair programming (Claude).
> All architecture decisions, debugging, and experimental results by the author.

---

## Results

| Model | Test Accuracy | Params |
|-------|-------------|--------|
| CNN only (baseline) | 98.49% | 689K |
| **CNN + GNN fusion** | **99.10%** | 805K |
| **Delta** | **+0.61%** | — |

The +0.61% improvement proves that modeling machine connectivity as a graph adds information beyond what signal analysis alone can capture.

---

## Architecture
```
Vibration signal (1×1024)  →  1D-CNN encoder  →  128-dim embedding ──┐
                                                                        ├──► Gated fusion → Fault class
Equipment graph (20 nodes) →  GraphSAGE GNN   →  128-dim embedding ──┘
```

**Equipment graph topology:**
- 20 nodes — one per bearing condition (fault type × severity × load)
- 90 edges — mechanical connections based on domain knowledge
- Edge types: severity propagation, shared shaft, baseline reference
- Avg node degree: 4.5

---

## Dataset

[CWRU Bearing Fault Dataset](https://engineering.case.edu/bearingdatacenter) — the standard benchmark for industrial fault detection.

| Class | Label | Description |
|-------|-------|-------------|
| Normal | 0 | No fault |
| Inner race | 1 | Crack on inner bearing ring |
| Ball | 2 | Defect on rolling ball |
| Outer race | 3 | Crack on outer bearing ring |

**Key design decision:** file-level train/val/test split to prevent data leakage from overlapping windows. Naive window-level splitting gives artificially inflated 100% accuracy.

---

## Project Structure
```
src/
├── data/
│   ├── cwru_dataset_v2.py     # file-level split dataset loader
│   └── graph_builder.py       # equipment graph construction
├── models/
│   ├── temporal_encoder.py    # 1D-CNN (689K params)
│   ├── gnn_encoder.py         # GraphSAGE (25K params)
│   └── fusion_model.py        # gated fusion (805K params)
└── train/
    ├── train_baseline_v2.py   # ablation A: CNN only
    └── train_fusion.py        # ablation B: CNN + GNN
app.py                         # Gradio demo
```

---

## Quickstart
```bash
# 1. Clone and setup
git clone https://github.com/Rothvichea/industrial-gnn-predictive-maintenance
cd industrial-gnn-predictive-maintenance
conda activate industrial-ai  # or your env with PyTorch + PyG

# 2. Get the dataset
git clone https://github.com/s-whynot/CWRU-dataset data/raw/CWRU

# 3. Train baseline (CNN only)
python src/train/train_baseline_v2.py

# 4. Train fusion model (CNN + GNN)
python src/train/train_fusion.py

# 5. Launch demo
python app.py  # → http://localhost:7860

# 6. View MLflow experiments
mlflow ui --port 5000  # → http://localhost:5000
```

---

## Environment

- Ubuntu 22.04, Python 3.10
- PyTorch 2.10 + CUDA 12.8
- PyTorch Geometric 2.7
- RTX 3060 Laptop GPU (6GB VRAM)
- Training time: ~5 min per model

---

## Why this matters for industrial AI

Most predictive maintenance models treat each machine in isolation. In real factories, machines are mechanically connected — a degrading motor bearing stresses connected pump bearings through the shaft. The GNN captures this propagation. The CNN alone cannot.

This project demonstrates the core skills required for applied ML research in industrial settings:
- Multimodal architecture design (signal + graph)
- Rigorous ablation study with honest baselines
- Domain-driven graph construction
- Reproducible ML pipelines with experiment tracking
- End-to-end demo deployment
