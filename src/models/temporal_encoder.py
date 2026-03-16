"""
1D-CNN Temporal Encoder for bearing fault signals.
Input:  (batch, 1, 1024)  — 1 channel, 1024 time steps
Output: (batch, embed_dim) — learned feature embedding
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class TemporalEncoder(nn.Module):
    """
    Hierarchical 1D-CNN with residual blocks.

    Input  → Stem(k=7, stride=2) → Stage1(64) → Stage2(128) → Stage3(256)
           → GlobalAvgPool → Linear(embed_dim)

    input (B,1,1024) → embedding (B, embed_dim)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResBlock1D(64),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            ResBlock1D(128),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            ResBlock1D(256),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)      # (B, 32,  512)
        x = self.stage1(x)    # (B, 64,  256)
        x = self.stage2(x)    # (B, 128,  64)
        x = self.stage3(x)    # (B, 256,  16)
        x = self.gap(x)       # (B, 256,   1)
        x = x.squeeze(-1)     # (B, 256)
        return self.projector(x)  # (B, embed_dim)


class FaultClassifier(nn.Module):
    """
    Standalone baseline: TemporalEncoder + classification head.
    Used for the no-GNN ablation experiment.
    """

    def __init__(self, num_classes: int = 4, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = TemporalEncoder(embed_dim=embed_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


if __name__ == "__main__":
    print("=== Temporal Encoder Smoke Test ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    enc = TemporalEncoder(embed_dim=128).to(device)
    x   = torch.randn(32, 1, 1024).to(device)
    emb = enc(x)
    print(f"Encoder:    input={tuple(x.shape)} → embedding={tuple(emb.shape)}")

    clf = FaultClassifier(num_classes=4, embed_dim=128).to(device)
    out = clf(x)
    print(f"Classifier: input={tuple(x.shape)} → logits={tuple(out.shape)}")

    n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    print("All OK!")
