import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(nn.Module):
    """
    Graph Convolutional Network (GCN) architecture for NOMA User Grouping.
    
    Layout:
    Input Node Features -> GCNConv -> ReLU -> Dropout -> GCNConv -> ReLU -> GCNConv -> Embedding -> Linear -> Softmax
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(GCNNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Embedding layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Final classification linear head
        self.classifier = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        h = x
        for i in range(self.num_layers):
            if edge_weight is not None and edge_weight.ndim > 1:
                weight = edge_weight.squeeze(-1)
            else:
                weight = edge_weight

            h = self.convs[i](h, edge_index, edge_weight=weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Learned node embedding representation
        node_embeddings = h

        # NOMA group prediction probabilities
        logits = self.classifier(node_embeddings)
        probs = F.softmax(logits, dim=-1)

        return probs, node_embeddings, logits
