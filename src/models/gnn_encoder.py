"""
GNN Encoder — GraphSAGE over the equipment graph.
Takes node features + graph structure → per-node embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """
    2-layer GraphSAGE encoder.

    Each bearing node aggregates information from its neighbours
    (connected bearings) to produce a context-aware embedding.

    Input:  node features (num_nodes, node_feat_dim)
    Output: node embeddings (num_nodes, embed_dim)
            OR graph embedding (1, embed_dim) if pooled
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        hidden_dim:    int = 64,
        embed_dim:     int = 128,
        dropout:       float = 0.3,
        num_layers:    int = 2,
    ):
        super().__init__()
        self.dropout   = dropout
        self.embed_dim = embed_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(SAGEConv(in_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor = None,
        pool:       bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:          (num_nodes, node_feat_dim)
            edge_index: (2, num_edges)
            batch:      (num_nodes,) — batch vector for pooling
            pool:       if True → return graph-level embedding via mean pool

        Returns:
            node embeddings: (num_nodes, embed_dim)
            OR graph embedding: (batch_size, embed_dim) if pool=True
        """
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if pool:
            # Global mean pool → single graph-level vector
            if batch is None:
                # Single graph — mean over all nodes
                x = x.mean(dim=0, keepdim=True)
            else:
                x = global_mean_pool(x, batch)

        return x


class GNNNodeClassifier(nn.Module):
    """
    GNN-only classifier for ablation B.
    Uses graph structure + node features only (no signal CNN).
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        hidden_dim:    int = 64,
        embed_dim:     int = 128,
        num_classes:   int = 4,
        dropout:       float = 0.3,
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        emb = self.encoder(x, edge_index, batch, pool=False)
        return self.head(emb)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.data.graph_builder import build_graph

    print("=== GNN Encoder Smoke Test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    graph = build_graph()
    x          = graph.x.to(device)
    edge_index = graph.edge_index.to(device)

    # Test node-level embeddings
    gnn = GNNEncoder(node_feat_dim=6, embed_dim=128).to(device)
    node_emb = gnn(x, edge_index, pool=False)
    print(f"Node embeddings: {tuple(node_emb.shape)}")

    # Test graph-level pooled embedding
    graph_emb = gnn(x, edge_index, pool=True)
    print(f"Graph embedding (pooled): {tuple(graph_emb.shape)}")

    # Test node classifier
    clf = GNNNodeClassifier(num_classes=4).to(device)
    logits = clf(x, edge_index)
    print(f"Node logits: {tuple(logits.shape)}")

    n_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    print("All OK!")
