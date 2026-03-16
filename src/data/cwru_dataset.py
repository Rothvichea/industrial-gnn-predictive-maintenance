"""
CWRU Bearing Dataset loader.
Reads raw .mat files, extracts DE (drive-end) signal,
slices into fixed windows, normalizes, returns labeled tensors.

Label map:
  0 = normal
  1 = inner race fault (IR)
  2 = ball fault (B)
  3 = outer race fault (OR)
"""

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict


# ── File manifest ─────────────────────────────────────────────────────────
# (relative_path, label, fault_type, severity_mil)
CWRU_MANIFEST = [
    # Normal (4 RPM conditions → all label 0)
    ("Normal/97_Normal_0.mat",  0, "normal", 0),
    ("Normal/98_Normal_1.mat",  0, "normal", 0),
    ("Normal/99_Normal_2.mat",  0, "normal", 0),
    ("Normal/100_Normal_3.mat", 0, "normal", 0),

    # Inner Race faults
    ("12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat", 1, "inner_race", 7),
    ("12k_Drive_End_Bearing_Fault_Data/IR/007/106_1.mat", 1, "inner_race", 7),
    ("12k_Drive_End_Bearing_Fault_Data/IR/014/169_0.mat", 1, "inner_race", 14),
    ("12k_Drive_End_Bearing_Fault_Data/IR/014/170_1.mat", 1, "inner_race", 14),
    ("12k_Drive_End_Bearing_Fault_Data/IR/021/209_0.mat", 1, "inner_race", 21),
    ("12k_Drive_End_Bearing_Fault_Data/IR/021/210_1.mat", 1, "inner_race", 21),

    # Ball faults
    ("12k_Drive_End_Bearing_Fault_Data/B/007/118_0.mat", 2, "ball", 7),
    ("12k_Drive_End_Bearing_Fault_Data/B/007/119_1.mat", 2, "ball", 7),
    ("12k_Drive_End_Bearing_Fault_Data/B/014/185_0.mat", 2, "ball", 14),
    ("12k_Drive_End_Bearing_Fault_Data/B/014/186_1.mat", 2, "ball", 14),
    ("12k_Drive_End_Bearing_Fault_Data/B/021/222_0.mat", 2, "ball", 21),
    ("12k_Drive_End_Bearing_Fault_Data/B/021/223_1.mat", 2, "ball", 21),

    # Outer Race faults (@ suffix in filenames)
    ("12k_Drive_End_Bearing_Fault_Data/OR/007/@6/130@6_0.mat", 3, "outer_race", 7),
    ("12k_Drive_End_Bearing_Fault_Data/OR/007/@6/131@6_1.mat", 3, "outer_race", 7),
    ("12k_Drive_End_Bearing_Fault_Data/OR/014/197@6_0.mat", 3, "outer_race", 14),
    ("12k_Drive_End_Bearing_Fault_Data/OR/021/@6/234_0.mat", 3, "outer_race", 21),
]

LABEL_NAMES = {0: "normal", 1: "inner_race", 2: "ball", 3: "outer_race"}


def _get_de_signal(mat: dict) -> np.ndarray:
    """Extract drive-end accelerometer signal from any CWRU .mat file."""
    for key in mat.keys():
        if key.startswith('_'):
            continue
        if 'DE_time' in key:
            return mat[key].flatten().astype(np.float64)
    raise KeyError(f"No DE_time key found. Available: {list(mat.keys())}")


def _get_rpm(mat: dict) -> int:
    for key in mat.keys():
        if 'RPM' in key.upper() and not key.startswith('_'):
            return int(mat[key].flatten()[0])
    return -1


def load_and_window(
    mat_path: str,
    label: int,
    window_size: int = 1024,
    stride: int = 512,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one .mat file and slice into overlapping windows.

    Returns:
        windows : (N, window_size) float32
        labels  : (N,)            int64
    """
    mat = sio.loadmat(mat_path)
    signal = _get_de_signal(mat)          # shape: (T,)

    # Slice into windows
    starts = range(0, len(signal) - window_size + 1, stride)
    windows = np.array([signal[s:s+window_size] for s in starts], dtype=np.float32)

    # Per-window zero-mean / unit-variance normalization
    if normalize:
        mean = windows.mean(axis=1, keepdims=True)
        std  = windows.std(axis=1, keepdims=True) + 1e-8
        windows = (windows - mean) / std

    labels = np.full(len(windows), label, dtype=np.int64)
    return windows, labels


class CWRUDataset(Dataset):
    """
    PyTorch Dataset for CWRU bearing fault classification.

    Args:
        root       : path to data/raw/CWRU/
        split      : 'train', 'val', or 'test'
        window_size: samples per window (default 1024 @ 12kHz = ~85ms)
        stride     : hop between windows (512 = 50% overlap)
        train_ratio: fraction of windows used for train
        val_ratio  : fraction used for val (rest = test)
        seed       : random seed for reproducible splits
    """

    def __init__(
        self,
        root: str = "data/raw/CWRU",
        split: str = "train",
        window_size: int = 1024,
        stride: int = 512,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.root        = Path(root)
        self.split       = split
        self.window_size = window_size
        self.label_names = LABEL_NAMES

        all_windows, all_labels = [], []
        self.skipped = []

        for rel_path, label, fault_type, severity in CWRU_MANIFEST:
            full_path = self.root / rel_path
            if not full_path.exists():
                self.skipped.append(str(full_path))
                continue
            try:
                windows, labels = load_and_window(
                    str(full_path), label, window_size, stride
                )
                all_windows.append(windows)
                all_labels.append(labels)
            except Exception as e:
                self.skipped.append(f"{full_path}: {e}")

        if not all_windows:
            raise RuntimeError("No .mat files loaded — check your data path.")

        X = np.concatenate(all_windows, axis=0)  # (N, window_size)
        y = np.concatenate(all_labels,  axis=0)  # (N,)

        # Reproducible stratified-ish split by shuffling then slicing
        rng   = np.random.default_rng(seed)
        idx   = rng.permutation(len(X))
        n     = len(idx)
        n_tr  = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == 'train':
            idx = idx[:n_tr]
        elif split == 'val':
            idx = idx[n_tr:n_tr+n_val]
        else:
            idx = idx[n_tr+n_val:]

        # Shape: (N, 1, window_size) — 1 channel for 1D-CNN
        self.X = torch.from_numpy(X[idx]).unsqueeze(1)
        self.y = torch.from_numpy(y[idx])

        self._print_summary()

    def _print_summary(self):
        unique, counts = torch.unique(self.y, return_counts=True)
        print(f"\n[CWRUDataset] split='{self.split}'  total={len(self.X)}")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"  label {u} ({LABEL_NAMES[u]}): {c} windows")
        if self.skipped:
            print(f"  skipped {len(self.skipped)} files")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def num_classes(self):
        return len(LABEL_NAMES)

    @property
    def input_channels(self):
        return 1

    @property
    def input_length(self):
        return self.window_size


if __name__ == "__main__":
    print("=== CWRU Dataset Smoke Test ===")
    for split in ['train', 'val', 'test']:
        ds = CWRUDataset(split=split)
        x, y = ds[0]
        print(f"  {split}: x.shape={x.shape}, y={y.item()}, "
              f"x.mean={x.mean():.4f}, x.std={x.std():.4f}")
    print("\nAll splits OK!")
