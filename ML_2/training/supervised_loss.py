import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedLoss(nn.Module):
    """
    Supervised Cross-Entropy Loss for labeled nodes.
    
    Ignores unlabeled nodes (label == -1).
    """
    def __init__(self):
        super(SupervisedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes mean Cross-Entropy loss on valid labeled samples.
        """
        labeled_mask = (targets != -1)
        if not labeled_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[labeled_mask]
        valid_targets = targets[labeled_mask]

        loss = self.ce_loss(valid_logits, valid_targets)
        return loss.mean()
