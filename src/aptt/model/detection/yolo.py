from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from aptt.heads.box import BBoxHead
from aptt.lightning_base.module import BaseModule
from aptt.loss.bbox import BboxLoss, RotatedBboxLoss
from aptt.model.backend_adapter import BackboneAdapter
from aptt.model.feature.fpn import FPN


class YOLO(nn.Module):
    def __init__(
        self,
        classification_model: nn.Module,
        backbone: BackboneAdapter,
        reg_max: int = 16,
        use_rotated_loss: bool = False,
        stage_indices: Sequence[int] = (3, 4, 5),
        extra_heads: dict[str, nn.Module] | None = None,
    ):
        """Initializes Yolo.

        Args:
            backbone (nn.Module): Feature extractor Backbone
            classification_model (nn.Module): Classification Model for direct classification
            reg_max (int): Maximum number of anchor boxes
            num_classes (int): Number of classes
            use_rotated_loss (bool): Whether to use rotated bounding box loss
            search_space (dict): Search space for hyperparameter optimization
            log_every_n_steps (int): Log metrics every n steps
            use_mlflow (bool): Whether to use MLflow for logging
            metrics (list): List of metrics to log
            stage_indices (Sequence[int]): Indizes der Feature Maps aus dem Backbone, die für die FPN verwendet werden.
        """
        super().__init__()

        # set backbone
        self.backbone = backbone
        self.backbone.set_stage_indices(stage_indices)

        # set featrure pyramide
        channels = self.get_backbone_out_channels(backbone)
        self.neck = FPN(channels, 256)

        # build head
        self.bbox_head = BBoxHead(256)  # BBox Prediction
        self.classification_head = classification_model  # Direktes Modell für Klassifikation

        # Wähle den passenden Bbox-Loss
        self.bbox_loss = RotatedBboxLoss(reg_max=reg_max) if use_rotated_loss else BboxLoss(reg_max=reg_max)

        self.extra_heads = nn.ModuleDict(extra_heads or {})

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of Bounding Box Predictions and Class Predictions

        Examples:
            >>> model = ObjectDetection(...)
            >>> model.forward(x)
        """
        x = args[0]

        features = self.backbone(x, return_stages=True)

        assert len(features) == len(self.neck.lateral_convs), (
            f"Backbone returned {len(features)} features, but FPN expected {len(self.neck.lateral_convs)}."
        )

        pyramid_feats = self.neck(features)

        outputs = {
            "bbox": self.bbox_head(pyramid_feats[-1]),
            "class": self.classification_head(pyramid_feats[-1]),
        }

        for name, head in self.extra_heads.items():
            outputs[name] = head(pyramid_feats[-1])

        return outputs

    @staticmethod
    def get_backbone_out_channels(backbone: nn.Module, x: torch.Tensor | None = None) -> list[int]:
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224) if x is None else x
            features = backbone(x, return_stages=True)
            return [f.shape[1] for f in features]

    @staticmethod
    def crop_from_rotated_boxes(
        images: torch.Tensor, boxes: torch.Tensor, size: tuple[int, int] = (224, 224)
    ) -> dict[str, Any]:
        device = images.device
        _b, _c, _h, _w = images.shape
        crops = []
        crop_indices = []
        xyxy_boxes = []

        for batch_idx, (img, img_boxes) in enumerate(zip(images, boxes)):
            for box_idx, box in enumerate(img_boxes):
                cx, cy, w, h, angle = box.tolist()

                scale_x = w / _w
                scale_y = h / _h
                tx = (2 * cx / _w) - 1
                ty = (2 * cy / _h) - 1

                theta = torch.tensor(
                    [
                        [scale_x * torch.cos(angle), -scale_y * torch.sin(angle), tx],
                        [scale_x * torch.sin(angle), scale_y * torch.cos(angle), ty],
                    ],
                    device=device,
                ).unsqueeze(0)

                grid = torch.nn.functional.affine_grid(theta, size=[1, _c, *size], align_corners=False)
                crop = torch.nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=False)
                crops.append(crop)
                crop_indices.append((batch_idx, box_idx))

                x1 = max(int(cx - w / 2), 0)
                y1 = max(int(cy - h / 2), 0)
                x2 = min(int(cx + w / 2), img.shape[2])
                y2 = min(int(cy + h / 2), img.shape[1])
                xyxy_boxes.append((batch_idx, box_idx, x1, y1, x2, y2))

        return {
            "crops": torch.cat(crops, dim=0),
            "indices": crop_indices,
            "boxes_xyxy": xyxy_boxes,
        }

    @staticmethod
    def crop_from_boxes(
        images: torch.Tensor, boxes: torch.Tensor, size: tuple[int, int] = (224, 224)
    ) -> dict[str, Any]:
        crops = []
        crop_indices = []
        xyxy_boxes = []

        for batch_idx, (img, img_boxes) in enumerate(zip(images, boxes)):
            for box_idx, box in enumerate(img_boxes):
                cx, cy, w, h, _ = box.tolist()

                x1 = max(int(cx - w / 2), 0)
                y1 = max(int(cy - h / 2), 0)
                x2 = min(int(cx + w / 2), img.shape[2])
                y2 = min(int(cy + h / 2), img.shape[1])

                crop = img[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=size, mode="bilinear", align_corners=False
                )
                crops.append(crop)
                crop_indices.append((batch_idx, box_idx))
                xyxy_boxes.append((batch_idx, box_idx, x1, y1, x2, y2))

        return {
            "crops": torch.cat(crops, dim=0),
            "indices": crop_indices,
            "boxes_xyxy": xyxy_boxes,
        }
