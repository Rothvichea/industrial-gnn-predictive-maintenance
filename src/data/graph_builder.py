"""
Graph construction for the industrial equipment graph.

We simulate a realistic factory topology:
- Each bearing in the dataset = one NODE
- Mechanical connections between bearings = EDGES
- Node features = tabular metadata (fault type history, severity, RPM, load)

Graph topology (domain knowledge):
  Motor shaft connects drive-end bearings across load conditions.
  Same fault type + adjacent severity = connected (fault can propagate).
  This gives us a meaningful graph structure to learn from.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple

# ── Node definitions ───────────────────────────────────────────────────────
# Each node = one bearing condition in the dataset
# (node_id, fault_type, severity, load_condition, rpm)
NODES = [
    # Normal bearings at 4 load conditions
    (0,  "normal",     0,  0, 1797),
    (1,  "normal",     0,  1, 1772),
    (2,  "normal",     0,  2, 1750),
    (3,  "normal",     0,  3, 1730),
    # Inner race faults — 3 severities x 2 load conditions
    (4,  "inner_race", 7,  0, 1797),
    (5,  "inner_race", 7,  1, 1772),
    (6,  "inner_race", 14, 0, 1797),
    (7,  "inner_race", 14, 1, 1772),
    (8,  "inner_race", 21, 0, 1797),
    (9,  "inner_race", 21, 1, 1772),
    # Ball faults — 3 severities x 2 load conditions
    (10, "ball",       7,  0, 1797),
    (11, "ball",       7,  1, 1772),
    (12, "ball",       14, 0, 1797),
    (13, "ball",       14, 1, 1772),
    (14, "ball",       21, 0, 1797),
    (15, "ball",       21, 1, 1772),
    # Outer race faults — 3 severities
    (16, "outer_race", 7,  0, 1797),
    (17, "outer_race", 7,  1, 1772),
    (18, "outer_race", 14, 0, 1797),
    (19, "outer_race", 21, 0, 1797),
]

FAULT_TYPE_MAP = {"normal": 0, "inner_race": 1, "ball": 2, "outer_race": 3}
NUM_NODES = len(NODES)


def build_node_features() -> torch.Tensor:
    """
    Build node feature matrix X of shape (NUM_NODES, num_features).

    Features per node:
      [0] fault_type_encoded  (0-3)
      [1] severity_normalized (0-1)
      [2] load_normalized     (0-1)
      [3] rpm_normalized      (0-1)
      [4] is_normal           (binary)
      [5] is_fault            (binary)
    """
    features = []
    max_severity = 21.0
    max_load     = 3.0
    max_rpm      = 1797.0

    for node_id, fault_type, severity, load, rpm in NODES:
        f = [
            FAULT_TYPE_MAP[fault_type] / 3.0,   # normalized fault type
            severity / max_severity,              # normalized severity
            load     / max_load,                  # normalized load
            rpm      / max_rpm,                   # normalized RPM
            1.0 if fault_type == "normal" else 0.0,
            0.0 if fault_type == "normal" else 1.0,
        ]
        features.append(f)

    return torch.tensor(features, dtype=torch.float32)


def build_edges() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index and edge_attr for the equipment graph.

    Edge rules (domain knowledge):
    1. Same fault type, adjacent severity → connected
       (fault propagation: severity 7 → 14 → 21)
    2. Same severity, different fault type at same load → connected
       (shared shaft: different fault modes at same operating condition)
    3. Normal nodes connect to all fault nodes at same load
       (baseline reference connections)

    Returns:
        edge_index: (2, num_edges) — COO format
        edge_attr:  (num_edges, 2) — [connection_type, weight]
    """
    edges = []
    edge_attrs = []

    node_map = {n[0]: n for n in NODES}

    for i, (nid_i, ft_i, sev_i, load_i, rpm_i) in enumerate(NODES):
        for j, (nid_j, ft_j, sev_j, load_j, rpm_j) in enumerate(NODES):
            if i >= j:
                continue

            connected    = False
            edge_type    = 0
            edge_weight  = 0.0

            # Rule 1: same fault type, adjacent severity, same load
            if (ft_i == ft_j and load_i == load_j and ft_i != "normal"):
                sev_diff = abs(sev_i - sev_j)
                if sev_diff in [7, 14]:   # adjacent severity levels
                    connected   = True
                    edge_type   = 1       # severity propagation
                    edge_weight = 1.0 - (sev_diff / 21.0)

            # Rule 2: same load, same severity → shared shaft condition
            if (load_i == load_j and sev_i == sev_j and ft_i != ft_j
                    and ft_i != "normal" and ft_j != "normal"):
                connected   = True
                edge_type   = 2           # shared operating condition
                edge_weight = 0.8

            # Rule 3: normal connects to faults at same load
            if (ft_i == "normal" and ft_j != "normal" and load_i == load_j):
                connected   = True
                edge_type   = 3           # baseline reference
                edge_weight = 0.5

            if connected:
                edges.append([i, j])
                edges.append([j, i])      # undirected → both directions
                edge_attrs.append([edge_type, edge_weight])
                edge_attrs.append([edge_type, edge_weight])

    edge_index = torch.tensor(edges,      dtype=torch.long).T   # (2, E)
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float32)  # (E, 2)
    return edge_index, edge_attr


def build_graph() -> Data:
    """Build the full equipment graph as a PyG Data object."""
    x          = build_node_features()
    edge_index, edge_attr = build_edges()

    graph = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        num_nodes  = NUM_NODES,
    )
    return graph


if __name__ == "__main__":
    print("=== Equipment Graph Smoke Test ===")
    graph = build_graph()
    print(f"Nodes:      {graph.num_nodes}")
    print(f"Edges:      {graph.num_edges} (undirected pairs x2)")
    print(f"Node feats: {graph.x.shape}")
    print(f"Edge attrs: {graph.edge_attr.shape}")
    print(f"\nNode feature sample (node 0 = normal):")
    print(f"  {graph.x[0].tolist()}")
    print(f"Node feature sample (node 8 = IR severity 21):")
    print(f"  {graph.x[8].tolist()}")

    # Validate graph
    assert graph.edge_index.max() < graph.num_nodes, "Invalid edge index!"
    assert not (graph.edge_index[0] == graph.edge_index[1]).any(), "Self-loops found!"
    print(f"\nGraph valid — no self-loops, all edges in range.")
    print(f"Avg degree: {graph.num_edges / graph.num_nodes:.1f}")
