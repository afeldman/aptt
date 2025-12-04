"""This module contains the BboxLoss class for computing training losses during training.

Author:
    Anton Feldmann <anton.feldmann@gmail.com>

Classes:
    BboxLoss: Criterion class for computing training losses during training.
    RotatedBboxLoss: Criterion class for computing training losses during training for rotated bounding boxes.
    AnkerloserBboxLoss: Ankerlose Bounding Box Loss-Funktion mit optionaler logarithmischer Skalierung.

Functions:
    forward: Compute the IoU loss and DFL loss.

"""

import torch
import torch.nn.functional as f
from loguru import logger
from torch import Tensor, nn

from aptt.loss.dfl import DFLoss
from aptt.metric.bbox_iou import bbox_iou, rotated_bbox_iou
from aptt.metric.probiou import probiou
from aptt.utils.bbox import bbox2dist
from aptt.utils.xy import xywh2xyxy


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training.

    This class computes the Intersection over Union (IoU) loss and the
    Distribution Focal Loss (DFL) for bounding box regression.

    Attributes:
        dfl_loss (DFLoss): Instance of the DFLoss class if `reg_max` > 1, else None.
    """

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.

        Args:
            reg_max (int): The maximum value for the regularization. Default is 16.
        """
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        target_bboxes: Tensor,
        target_scores: Tensor,
        target_scores_sum: float,
        fg_mask: Tensor,  # Korrektur: fg_mask ist ein Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute the IoU loss and DFL loss.

        Args:
            pred_dist (Tensor): Predicted distances.
            pred_bboxes (Tensor): Predicted bounding boxes.
            anchor_points (Tensor): Anchor points.
            target_bboxes (Tensor): Target bounding boxes.
            target_scores (Tensor): Target scores.
            target_scores_sum (float): Sum of target scores.
            fg_mask (Tensor): Foreground mask.

        Returns:
            Tuple[Tensor, Tensor]: IoU loss and DFL loss.
        """
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_dist.device), torch.tensor(0.0, device=pred_dist.device)

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / max(target_scores_sum, 1e-6)  # Schutz gegen Division durch 0

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / max(target_scores_sum, 1e-6)
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training for rotated bounding boxes."""

    def forward(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        target_bboxes: Tensor,
        target_scores: Tensor,
        target_scores_sum: float,
        fg_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the IoU loss and DFL loss for rotated bounding boxes."""
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_dist.device), torch.tensor(0.0, device=pred_dist.device)

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / max(target_scores_sum, 1e-6)

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / max(target_scores_sum, 1e-6)
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class AnkerloserBboxLoss(nn.Module):
    """Ankerlose Bounding Box Loss-Funktion mit optionaler logarithmischer Skalierung."""

    def __init__(self, log_reg: bool = False, use_rotated_iou: bool = True, reduction: str='mean'):
        super().__init__()
        self.log_reg = log_reg
        self.use_rotated_iou = use_rotated_iou
        self.reduction = reduction

    def forward(
        self, bbox_pred: Tensor, target_bboxes: Tensor, weights: tuple[float, float, float] = (0.4, 0.2, 0.4)
    ) -> Tensor:
        """Berechnet den kombinierten Verlust für ankerlose Bounding Box Regression."""
        # Falls `weights` nicht genau 3 Werte hat, Standardwerte setzen
        if len(weights) != 3:
            logger.warning(f"⚠️ `weights` hat {len(weights)} Werte. Ersetze mit Standardwerten (0.4, 0.2, 0.4).")
            weights = (0.4, 0.2, 0.4)

        if bbox_pred.shape[-1] == 4:
            if self.use_rotated_iou:
                logger.warning("⚠️ Rotated IoU aktiv, aber Bounding Boxes haben kein Theta! Fallback auf normalen IoU.")
            loss_theta = 0.0
            self.use_rotated_iou = False

        # Falls die Summe der Gewichte nicht 1.0 ist, normalisieren
        weight_sum = sum(weights)
        norm_weights = [float(w / weight_sum) for w in weights] if weight_sum != 1.0 else weights

        # 📌 **1. L1-Regressionsverlust für W, H & θ**
        if self.log_reg:
            loss_wh = f.smooth_l1_loss(
                torch.log1p(bbox_pred[..., 2:4]),
                torch.log1p(target_bboxes[..., 2:4]),
                reduction=self.reduction,
            )
        else:
            loss_wh = f.smooth_l1_loss(bbox_pred[..., 2:4], target_bboxes[..., 2:4], reduction=self.reduction)

        # 🔄 **L1-Verlust für die Rotation (θ)**
        if bbox_pred.shape[-1] == 4:  # Kein θ vorhanden
            loss_theta = 0.0
        else:
            loss_theta = f.smooth_l1_loss(bbox_pred[..., 4], target_bboxes[..., 4], reduction=self.reduction)

        # 📌 **2. IoU-Verlust für Rotation oder Standard-IoU**
        loss_iou = (
            1 - rotated_bbox_iou(bbox_pred, target_bboxes)
            if self.use_rotated_iou
            else 1 - bbox_iou(bbox_pred, target_bboxes, xywh=True)
        )
        loss_iou = loss_iou.mean()

        # 📌 **3. Gesamtverlust mit normalisierten Gewichtungen**
        loss = norm_weights[0] * loss_wh + norm_weights[1] * loss_theta + norm_weights[2] * loss_iou

        return loss
