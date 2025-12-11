r"""Distribution Focal Loss (DFL) module.
-------------------------------------

This module contains the Distribution Focal Loss (DFL) class for computing DFL losses during training.

.. math::
DFL = \sum_{i=1}^{N} (CE(p_{i}, t_{i}) * w_{
i} + CE(p_{i}, t_{i} + 1) * (1 - w_{i})) / N

Example:
>>> import torch
>>> from aptt.loss import DFLoss
>>> criterion = DFLoss()
>>> preds = torch.randn(2, 16)
>>> target = torch.randint(0, 16, (2,))
>>> loss = criterion(preds, target)
>>> loss
tensor(2.4175)
>>> loss_items
tensor(2.4175)

Attributes:
DFLoss: The class for computing DFL losses.

Methods:
__call__: Compute the DFL loss between predictions and true labels.

References:
https://ieeexplore.ieee.org/document/9792391

"""

from torch import Tensor, nn
from torch.nn import functional as F


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training.

    This class implements the Distribution Focal Loss (DFL) as proposed in the
    Generalized Focal Loss paper.

    Attributes:
        reg_max (int): The maximum value for the regularization.
    """

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module.

        Args:
            reg_max (int): The maximum value for the regularization. Default is 16.
        """
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        """Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391

        Args:
            pred_dist (Tensor): Predicted distances.
            target (Tensor): Target distances.

        Returns:
            Tensor: Computed DFL loss.
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(
            tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(
                    tl.shape) * wr).mean(-1, keepdim=True)
