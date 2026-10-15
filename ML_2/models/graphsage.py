import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGENet(nn.Module):
    """
    GraphSAGE architecture for dynamic NOMA User Grouping.
    
    Layout:
    Input Node Features -> GraphSAGE Layer -> ReLU -> Dropout -> GraphSAGE Layer -> ReLU -> GraphSAGE Layer -> Embedding -> Linear -> Softmax
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(GraphSAGENet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Embedding layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))

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
