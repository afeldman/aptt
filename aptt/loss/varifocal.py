import torch
from torch import nn
from torch.nn import functional as F

from aptt.utils.autocast import autocast


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score: torch.Tensor,
                gt_score: torch.Tensor,
                label: torch.Tensor,
                alpha: float = 0.75,
                gamma: float = 2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (
            1 - label) + gt_score * label

        with autocast(enable=False):
            loss = ((F.binary_cross_entropy_with_logits(
                pred_score.float(), gt_score.float(), reduction="none") *
                     weight).mean(1).sum())
        return loss
