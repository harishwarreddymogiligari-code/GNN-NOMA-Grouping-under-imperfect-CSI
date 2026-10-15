import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ConfidenceUncertaintyPseudoLabeler(nn.Module):
    """
    Confidence-Aware & CSI Uncertainty-Filtered Pseudo-Labeling Engine.
    
    Filters unlabeled samples based on dual criteria:
    1. Prediction Confidence: max(P(y_i)) >= CONFIDENCE_THRESHOLD
    2. CSI Uncertainty: csi_uncertainty_i <= UNCERTAINTY_THRESHOLD
    """
    def __init__(
        self,
        confidence_threshold: float = 0.90,
        uncertainty_threshold: float = 0.50
    ):
        super(ConfidenceUncertaintyPseudoLabeler, self).__init__()
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def select_pseudo_labels(
        self,
        probs: torch.Tensor,
        csi_uncertainty: torch.Tensor,
        unlabeled_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Identifies reliable pseudo-labels for unlabeled nodes.
        
        Returns:
            pseudo_labels: Tensor of predicted class indices
            valid_mask: Boolean tensor marking selected reliable samples
            stats: Dictionary of selection coverage statistics
        """
        max_probs, predicted_labels = torch.max(probs, dim=-1)

        # Confidence criteria
        conf_mask = (max_probs >= self.confidence_threshold)

        # CSI Uncertainty criteria (lower uncertainty is more reliable)
        # Normalize uncertainty dynamically if values are outside [0, 1]
        if csi_uncertainty is not None and csi_uncertainty.numel() > 0:
            uncert_vals = csi_uncertainty.squeeze()
            if uncert_vals.max() > 1.0:
                # Use percentile thresholding
                threshold_val = torch.quantile(uncert_vals, self.uncertainty_threshold)
            else:
                threshold_val = self.uncertainty_threshold
            uncert_mask = (uncert_vals <= threshold_val)
        else:
            uncert_mask = torch.ones_like(conf_mask)

        # Combine all criteria on unlabeled nodes
        valid_pseudo_mask = unlabeled_mask & conf_mask & uncert_mask

        total_unlabeled = int(unlabeled_mask.sum().item())
        total_pseudo = int(valid_pseudo_mask.sum().item())
        coverage = total_pseudo / total_unlabeled if total_unlabeled > 0 else 0.0

        stats = {
            "total_unlabeled": total_unlabeled,
            "selected_pseudo": total_pseudo,
            "coverage": coverage
        }

        return predicted_labels, valid_pseudo_mask, stats

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        csi_uncertainty: torch.Tensor,
        unlabeled_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes Pseudo-Label Cross-Entropy Loss on filtered reliable unlabeled samples.
        """
        if unlabeled_mask is None or not unlabeled_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True), {"coverage": 0.0}

        pseudo_labels, valid_mask, stats = self.select_pseudo_labels(probs, csi_uncertainty, unlabeled_mask)

        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True), stats

        selected_logits = logits[valid_mask]
        selected_targets = pseudo_labels[valid_mask]

        loss = self.ce_loss(selected_logits, selected_targets).mean()
        return loss, stats
