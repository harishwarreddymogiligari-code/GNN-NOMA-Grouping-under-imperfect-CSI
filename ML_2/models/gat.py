import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(nn.Module):
    """
    Graph Attention Network (GAT) architecture for dynamic NOMA User Grouping.
    
    Uses multi-head attention mechanism across user graph nodes.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 2
    ):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))

        # Embedding layer (concat=False to output hidden_dim)
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))

        # Final classification linear head
        self.classifier = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Learned node embedding representation
        node_embeddings = h

        # NOMA group prediction probabilities
        logits = self.classifier(node_embeddings)
        probs = F.softmax(logits, dim=-1)

        return probs, node_embeddings, logits
