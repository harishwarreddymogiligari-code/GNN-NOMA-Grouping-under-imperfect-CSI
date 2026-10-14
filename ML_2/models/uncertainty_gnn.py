import torch
import torch.nn as nn
from typing import Tuple
from .gcn import GCNNet
from .graphsage import GraphSAGENet
from .gat import GATNet


class UncertaintyAwareGNN(nn.Module):
    """
    Uncertainty-Aware Graph Neural Network for Dynamic NOMA User Grouping.
    
    Explicitly incorporates CSI uncertainty as a node feature and routes
    graph structured user attributes through a configurable GNN backbone
    (GraphSAGE, GCN, or GAT).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        gnn_type: str = "GraphSAGE",
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(UncertaintyAwareGNN, self).__init__()
        self.gnn_type = gnn_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        gnn_type_upper = gnn_type.upper()
        if gnn_type_upper == "GRAPHSAGE" or gnn_type_upper == "SAGE":
            self.backbone = GraphSAGENet(in_channels, hidden_dim, out_channels, num_layers, dropout)
        elif gnn_type_upper == "GCN":
            self.backbone = GCNNet(in_channels, hidden_dim, out_channels, num_layers, dropout)
        elif gnn_type_upper == "GAT":
            self.backbone = GATNet(in_channels, hidden_dim, out_channels, num_layers, dropout)
        else:
            raise ValueError(f"Unsupported GNN type: '{gnn_type}'. Supported: 'GraphSAGE', 'GCN', 'GAT'.")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (group_probs, node_embeddings, logits).
        """
        return self.backbone(x, edge_index, edge_weight=edge_attr)
