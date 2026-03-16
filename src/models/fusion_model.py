import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from src.models.temporal_encoder import TemporalEncoder
from src.models.gnn_encoder import GNNEncoder


class FusionModel(nn.Module):
    def __init__(self, node_feat_dim=6, cnn_embed_dim=128, gnn_embed_dim=128, num_classes=4, dropout=0.3):
        super().__init__()
        self.cnn  = TemporalEncoder(in_channels=1, embed_dim=cnn_embed_dim, dropout=dropout)
        self.gnn  = GNNEncoder(node_feat_dim=node_feat_dim, hidden_dim=64, embed_dim=gnn_embed_dim, dropout=dropout)
        fusion_dim = cnn_embed_dim + gnn_embed_dim
        self.gate = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, signal, node_ids, graph_x, edge_index):
        cnn_emb  = self.cnn(signal)                            # (B, 128)
        node_emb = self.gnn(graph_x, edge_index, pool=False)  # (N, 128)
        gnn_emb  = node_emb[node_ids]                         # (B, 128)
        fused    = torch.cat([cnn_emb, gnn_emb], dim=-1)      # (B, 256)
        fused    = self.gate(fused) * fused                    # gated
        return self.head(fused)                                # (B, num_classes)


if __name__ == "__main__":
    from src.data.graph_builder import build_graph

    print("=== Fusion Model Smoke Test ===")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    graph      = build_graph()
    graph_x    = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    model      = FusionModel(num_classes=4).to(device)

    B        = 32
    signal   = torch.randn(B, 1, 1024).to(device)
    node_ids = torch.randint(0, 20, (B,)).to(device)
    logits   = model(signal, node_ids, graph_x, edge_index)

    print(f"Input signal:  {tuple(signal.shape)}")
    print(f"Output logits: {tuple(logits.shape)}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnn_p = sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)
    gnn_p = sum(p.numel() for p in model.gnn.parameters() if p.requires_grad)
    print(f"\nParam breakdown:")
    print(f"  CNN encoder: {cnn_p:,}")
    print(f"  GNN encoder: {gnn_p:,}")
    print(f"  Total:       {total:,}")
    print("All OK!")
