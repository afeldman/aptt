"""Centernet module."""

from torch import Tensor, norm, stack, tensor, topk, zeros
from torch.nn import Module
import torch.nn.functional as F
from torchvision.ops import batched_nms

from aptt.heads.centernet import MultiScaleCenterNetHead
from aptt.model.feature.fpn import FPN


def topk_heatmap(heatmap: Tensor, k: int = 100):
    """Args:
        heatmap: Tensor of shape (B, C, H, W)
        k: number of top scores to extract

    Returns:
        scores: (B, k)
        indices: (B, k)
        classes: (B, k)
        ys: (B, k)
        xs: (B, k)
    """
    B, C, H, W = heatmap.shape
    heatmap = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1) * (
        heatmap == heatmap.max(dim=1, keepdim=True)[0]
    )

    # flatten heatmap: (B, C*H*W)
    heatmap = heatmap.view(B, C, -1)
    topk_scores, topk_inds = topk(heatmap, k)  # (B, C, k)

    topk_scores = topk_scores.view(B, -1)  # (B, C*k)
    topk_inds = topk_inds.view(B, -1)  # (B, C*k)

    topk_classes = topk_inds // (H * W)  # (B, C*k)
    topk_inds = topk_inds % (H * W)

    topk_ys = (topk_inds // W).float()
    topk_xs = (topk_inds % W).float()

    return topk_scores, topk_classes, topk_ys, topk_xs


def match_triplets(tl_coords, br_coords, ct_coords, max_center_dist=2.0):
    """Filtert tl/br Paare durch Vergleich mit ct-Kandidaten.

    Args:
        tl_coords: Tensor (N, 2) - (x, y) top-left
        br_coords: Tensor (N, 2) - (x, y) bottom-right
        ct_coords: Tensor (M, 2) - mögliche center keypoints

    Returns:
        List[Tuple[tl_idx, br_idx]]: gültige Triplet-Indizes
    """
    matched = []
    for i, (tl_x, tl_y) in enumerate(tl_coords):
        for j, (br_x, br_y) in enumerate(br_coords):
            if br_x <= tl_x or br_y <= tl_y:
                continue  # ungültige Box
            cx = (tl_x + br_x) / 2
            cy = (tl_y + br_y) / 2
            # prüfe, ob ein ct nahe dem center liegt
            center = tensor([cx, cy], device=ct_coords.device)
            dists = norm(ct_coords - center, dim=1)
            if (dists < max_center_dist).any():
                matched.append((i, j))
    return matched


def apply_nms(boxes: Tensor, iou_thresh: float = 0.5) -> Tensor:
    if boxes.numel() == 0:
        return boxes
    coords = boxes[:, :4]
    scores = boxes[:, 4]
    classes = boxes[:, 5]
    keep = batched_nms(coords, scores, classes, iou_thresh)
    return boxes[keep]


def rescale_boxes(
    boxes: Tensor, input_size: tuple[int, int], output_size: tuple[int, int]
) -> Tensor:
    h_in, w_in = input_size
    h_out, w_out = output_size
    scale_x = w_in / w_out
    scale_y = h_in / h_out
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes


class CenterNetModel(Module):
    def __init__(
        self,
        backbone,
        in_channels_list,
        fpn_out_channels=256,
        num_classes=1,
        use_bbox=True,
        do_rescale=True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = FPN(in_channels_list, fpn_out_channels)
        self.head = MultiScaleCenterNetHead(
            [fpn_out_channels] * len(in_channels_list), num_classes=num_classes, use_bbox=use_bbox
        )
        self.do_rescale = do_rescale

    def forward(self, x):
        features = self.backbone(x)  # [C3, C4, C5]
        pyramid = self.neck(features)  # [P3, P4, P5]
        outputs = self.head(pyramid)  # list of dicts
        return outputs


class CenterNetDecoder(Module):
    def __init__(self, topk=100, score_thresh=0.1, do_rescale=True) -> None:
        super().__init__()
        self.topk = topk
        self.score_thresh = score_thresh
        self.do_rescale = do_rescale

    def forward(self, outputs, img_size):
        B = outputs[0]["tl_heat"].shape[0]
        all_boxes = []

        for b in range(B):
            all_scale_boxes = []

            for scale_out in outputs:
                tl_scores, tl_classes, tl_ys, tl_xs = topk_heatmap(
                    scale_out["tl_heat"][b : b + 1], self.topk
                )
                br_scores, _, br_ys, br_xs = topk_heatmap(
                    scale_out["br_heat"][b : b + 1], self.topk
                )
                _, _, ct_ys, ct_xs = topk_heatmap(scale_out["ct_heat"][b : b + 1], self.topk)

                tl_coords = stack([tl_xs[0], tl_ys[0]], dim=1)
                br_coords = stack([br_xs[0], br_ys[0]], dim=1)
                ct_coords = stack([ct_xs[0], ct_ys[0]], dim=1)

                pairs = match_triplets(tl_coords, br_coords, ct_coords)

                for i, j in pairs:
                    x1, y1 = tl_coords[i]
                    x2, y2 = br_coords[j]
                    score = (tl_scores[0][i] + br_scores[0][j]) / 2
                    cls = tl_classes[0][i]
                    if score >= self.score_thresh:
                        all_scale_boxes.append(
                            tensor([x1, y1, x2, y2, score, cls], device=tl_coords.device)
                        )

                if "bbox" in scale_out:
                    reg = scale_out["bbox"][b]
                    for i, j in pairs:
                        center_x = (tl_coords[i][0] + br_coords[j][0]) / 2
                        center_y = (tl_coords[i][1] + br_coords[j][1]) / 2
                        cxi = int(center_x.item())
                        cyi = int(center_y.item())
                        box_reg = reg[:, cyi, cxi]
                        pred_cx, pred_cy, w, h = box_reg
                        x1 = pred_cx - w / 2
                        y1 = pred_cy - h / 2
                        x2 = pred_cx + w / 2
                        y2 = pred_cy + h / 2
                        score = (tl_scores[0][i] + br_scores[0][j]) / 2
                        cls = tl_classes[0][i]
                        if score >= self.score_thresh:
                            all_scale_boxes.append(
                                tensor([x1, y1, x2, y2, score, cls], device=reg.device)
                            )

            if all_scale_boxes:
                boxes = stack(all_scale_boxes)
            else:
                boxes = zeros((0, 6), device=outputs[0]["tl_heat"].device)

            heatmap_size = scale_out["tl_heat"].shape[2:]
            if self.do_rescale:
                boxes = rescale_boxes(boxes, img_size, heatmap_size)

            all_boxes.append(boxes)

        return all_boxes
