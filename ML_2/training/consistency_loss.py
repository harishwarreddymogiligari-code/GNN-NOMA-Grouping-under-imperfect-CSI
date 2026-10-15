import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def perturb_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor = None,
    edge_drop_rate: float = 0.15,
    feature_mask_rate: float = 0.10,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates a stochastic perturbation of a graph by randomly dropping edges
    and applying feature noise / masking to produce augmented graph views.
    """
    if not training or (edge_drop_rate <= 0 and feature_mask_rate <= 0):
        return x, edge_index, edge_attr

    num_edges = edge_index.size(1)
    if num_edges > 0 and edge_drop_rate > 0:
        edge_mask = torch.rand(num_edges, device=x.device) > edge_drop_rate
        # Ensure at least 1 edge remains if non-empty
        if not edge_mask.any():
            edge_mask[0] = True
        perturbed_edge_index = edge_index[:, edge_mask]
        perturbed_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    else:
        perturbed_edge_index = edge_index
        perturbed_edge_attr = edge_attr

    if feature_mask_rate > 0:
        feat_mask = torch.rand_like(x) > feature_mask_rate
        # Multiply feature matrix by binary mask with scaling adjustment
        perturbed_x = x * feat_mask.float()
    else:
        perturbed_x = x

    return perturbed_x, perturbed_edge_index, perturbed_edge_attr


class GraphPerturbedConsistencyLoss(nn.Module):
    """
    Consistency Regularization Loss across dual perturbed graph views (Graph A & Graph B).
    
    Computes KL Divergence / MSE between P(A) and P(B) distributions on unlabeled nodes.
    """
    def __init__(
        self,
        edge_drop_rate: float = 0.15,
        feature_mask_rate: float = 0.10,
        metric: str = "kl"
    ):
        super(GraphPerturbedConsistencyLoss, self).__init__()
        self.edge_drop_rate = edge_drop_rate
        self.feature_mask_rate = feature_mask_rate
        self.metric = metric

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        unlabeled_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes consistency loss between predictions from Graph A and Graph B.
        """
        if unlabeled_mask is None or not unlabeled_mask.any():
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Generate View A
        xA, edge_idxA, edge_attrA = perturb_graph(
            x, edge_index, edge_attr, self.edge_drop_rate, self.feature_mask_rate, training=model.training
        )
        probsA, _, _ = model(xA, edge_idxA, edge_attrA)

        # Generate View B
        xB, edge_idxB, edge_attrB = perturb_graph(
            x, edge_index, edge_attr, self.edge_drop_rate, self.feature_mask_rate, training=model.training
        )
        probsB, _, _ = model(xB, edge_idxB, edge_attrB)

        pA = torch.clamp(probsA[unlabeled_mask], min=1e-7, max=1.0)
        pB = torch.clamp(probsB[unlabeled_mask], min=1e-7, max=1.0)

        if self.metric == "kl":
            log_pA = torch.log(pA)
            log_pB = torch.log(pB)
            kl_AB = F.kl_div(log_pB, pA, reduction='batchmean')
            kl_BA = F.kl_div(log_pA, pB, reduction='batchmean')
            loss = 0.5 * (kl_AB + kl_BA)
        else:
            loss = F.mse_loss(pA, pB)

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        return loss
