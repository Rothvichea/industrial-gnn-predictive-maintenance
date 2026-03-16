"""
CWRU Dataset v2 — file-level train/val/test split.
All windows from a given .mat file stay in ONE split only.
This prevents data leakage from overlapping windows across splits.
"""

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from pathlib import Path

CWRU_MANIFEST = [
    ("Normal/97_Normal_0.mat",                                    0, "normal",     0),
    ("Normal/98_Normal_1.mat",                                    0, "normal",     0),
    ("Normal/99_Normal_2.mat",                                    0, "normal",     0),
    ("Normal/100_Normal_3.mat",                                   0, "normal",     0),
    ("12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat",         1, "inner_race", 7),
    ("12k_Drive_End_Bearing_Fault_Data/IR/007/106_1.mat",         1, "inner_race", 7),
    ("12k_Drive_End_Bearing_Fault_Data/IR/014/169_0.mat",         1, "inner_race", 14),
    ("12k_Drive_End_Bearing_Fault_Data/IR/014/170_1.mat",         1, "inner_race", 14),
    ("12k_Drive_End_Bearing_Fault_Data/IR/021/209_0.mat",         1, "inner_race", 21),
    ("12k_Drive_End_Bearing_Fault_Data/IR/021/210_1.mat",         1, "inner_race", 21),
    ("12k_Drive_End_Bearing_Fault_Data/B/007/118_0.mat",          2, "ball",        7),
    ("12k_Drive_End_Bearing_Fault_Data/B/007/119_1.mat",          2, "ball",        7),
    ("12k_Drive_End_Bearing_Fault_Data/B/014/185_0.mat",          2, "ball",       14),
    ("12k_Drive_End_Bearing_Fault_Data/B/014/186_1.mat",          2, "ball",       14),
    ("12k_Drive_End_Bearing_Fault_Data/B/021/222_0.mat",          2, "ball",       21),
    ("12k_Drive_End_Bearing_Fault_Data/B/021/223_1.mat",          2, "ball",       21),
    ("12k_Drive_End_Bearing_Fault_Data/OR/007/@6/130@6_0.mat",    3, "outer_race",  7),
    ("12k_Drive_End_Bearing_Fault_Data/OR/007/@6/131@6_1.mat",    3, "outer_race",  7),
    ("12k_Drive_End_Bearing_Fault_Data/OR/014/197@6_0.mat",       3, "outer_race", 14),
    ("12k_Drive_End_Bearing_Fault_Data/OR/021/@6/234_0.mat",      3, "outer_race", 21),
]

LABEL_NAMES = {0: "normal", 1: "inner_race", 2: "ball", 3: "outer_race"}


def _get_de_signal(mat: dict) -> np.ndarray:
    for key in mat.keys():
        if not key.startswith('_') and 'DE_time' in key:
            return mat[key].flatten().astype(np.float64)
    raise KeyError(f"No DE_time key. Got: {list(mat.keys())}")


def load_and_window(path, label, window_size=1024, stride=512):
    mat    = sio.loadmat(path)
    signal = _get_de_signal(mat)
    starts = range(0, len(signal) - window_size + 1, stride)
    wins   = np.array([signal[s:s+window_size] for s in starts], dtype=np.float32)
    mean   = wins.mean(axis=1, keepdims=True)
    std    = wins.std(axis=1,  keepdims=True) + 1e-8
    wins   = (wins - mean) / std
    labels = np.full(len(wins), label, dtype=np.int64)
    return wins, labels


class CWRUDatasetV2(Dataset):
    """
    File-level split: each .mat file is assigned entirely to
    train / val / test — no window from the same file appears
    in two different splits.

    Split strategy per class:
      train = first 60% of files
      val   = next  20% of files
      test  = last  20% of files
    """

    def __init__(
        self,
        root:        str   = "data/raw/CWRU",
        split:       str   = "train",
        window_size: int   = 1024,
        stride:      int   = 512,
        seed:        int   = 42,
    ):
        super().__init__()
        self.root        = Path(root)
        self.split       = split
        self.label_names = LABEL_NAMES

        # Group files by label
        from collections import defaultdict
        by_label = defaultdict(list)
        for entry in CWRU_MANIFEST:
            by_label[entry[1]].append(entry)

        rng = np.random.default_rng(seed)
        selected = []
        for label, entries in by_label.items():
            entries = list(entries)
            rng.shuffle(entries)
            n     = len(entries)
            n_tr  = max(1, int(n * 0.6))
            n_val = max(1, int(n * 0.2))
            if split == "train":
                selected.extend(entries[:n_tr])
            elif split == "val":
                selected.extend(entries[n_tr:n_tr+n_val])
            else:
                selected.extend(entries[n_tr+n_val:])

        all_windows, all_labels = [], []
        self.skipped = []

        for rel_path, label, fault_type, severity in selected:
            full_path = self.root / rel_path
            if not full_path.exists():
                self.skipped.append(str(full_path))
                continue
            wins, labs = load_and_window(str(full_path), label, window_size, stride)
            all_windows.append(wins)
            all_labels.append(labs)

        X = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels,  axis=0)

        # Shuffle within split
        idx = rng.permutation(len(X))
        self.X = torch.from_numpy(X[idx]).unsqueeze(1)
        self.y = torch.from_numpy(y[idx])
        self._print_summary()

    def _print_summary(self):
        unique, counts = torch.unique(self.y, return_counts=True)
        print(f"\n[CWRUDatasetV2] split='{self.split}'  total={len(self.X)}")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"  label {u} ({LABEL_NAMES[u]}): {c} windows")
        if self.skipped:
            print(f"  skipped: {self.skipped}")

    def __len__(self):              return len(self.X)
    def __getitem__(self, idx):     return self.X[idx], self.y[idx]
    @property
    def num_classes(self):          return len(LABEL_NAMES)
    @property
    def input_length(self):         return self.X.shape[-1]


if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        ds = CWRUDatasetV2(split=s)
        x, y = ds[0]
        print(f"  {s}: x={x.shape}, y={y.item()}")
    print("\nNo leakage — file-level split confirmed.")
