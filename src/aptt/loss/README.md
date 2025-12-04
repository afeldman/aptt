# Loss Functions

Verlustfunktionen für verschiedene Deep Learning Tasks.

## Module

### Language Models

- **`classification.py`** - Cross-Entropy und Focal Loss für Classification
- **`mtp_loss.py`** - Multi-Token Prediction Loss für DeepSeek-V3
- **`distill.py`** - Knowledge Distillation Loss
- **`lwf.py`** - Learning without Forgetting Loss

### Object Detection

- **`detection.py`** - Kombinierte Detection Loss (Classification + Localization)
- **`bbox.py`** - Bounding Box Losses (GIoU, DIoU, CIoU)
- **`centernet.py`** - CenterNet Heatmap Loss
- **`dfl.py`** - Distribution Focal Loss für Box Regression
- **`focal.py`** - Focal Loss für Class Imbalance
- **`varifocal.py`** - VariFocal Loss (ICCV 2021)

### Keypoint & Segmentation

- **`keypoint.py`** - Keypoint Detection Loss
- **`heat.py`** - Heatmap Regression Loss
- **`segmentation.py`** - Segmentation Losses (Dice, IoU)

### Audio

- **`mel.py`** - Mel-Spectrogram Loss
- **`snr.py`** - Signal-to-Noise Ratio Loss
- **`rmse.py`** - Root Mean Squared Error

## Verwendung

### Cross-Entropy Loss

```python
from aptt.loss import CrossEntropyLoss

loss_fn = CrossEntropyLoss(
    ignore_index=-100,
    label_smoothing=0.1
)

loss = loss_fn(logits, targets)
```

### Multi-Token Prediction Loss

```python
from aptt.loss import MTPLoss

mtp_loss = MTPLoss(
    mtp_depth=3,
    mtp_lambda=0.3  # Gewicht für MTP-Terme
)

loss = mtp_loss(logits_list, targets)
```

### Bounding Box Loss

```python
from aptt.loss import GIoULoss, DIoULoss, CIoULoss

# Generalized IoU
giou_loss = GIoULoss()
loss = giou_loss(pred_boxes, target_boxes)

# Distance IoU (berücksichtigt Zentrumsabstand)
diou_loss = DIoULoss()
loss = diou_loss(pred_boxes, target_boxes)

# Complete IoU (+ Aspect Ratio)
ciou_loss = CIoULoss()
loss = ciou_loss(pred_boxes, target_boxes)
```

### Focal Loss

```python
from aptt.loss import FocalLoss

focal_loss = FocalLoss(
    alpha=0.25,      # Klassen-Gewichtung
    gamma=2.0,       # Fokus auf schwere Beispiele
    reduction='mean'
)

loss = focal_loss(logits, targets)
```

### CenterNet Loss

```python
from aptt.loss import CenterNetLoss

centernet_loss = CenterNetLoss(
    hm_weight=1.0,   # Heatmap Loss
    wh_weight=0.1,   # Width/Height Loss
    reg_weight=1.0   # Offset Regression Loss
)

loss = centernet_loss(predictions, targets)
```

### Knowledge Distillation

```python
from aptt.loss import DistillationLoss

distill_loss = DistillationLoss(
    temperature=4.0,
    alpha=0.5  # Balance zwischen Hard und Soft Targets
)

loss = distill_loss(
    student_logits=student_out,
    teacher_logits=teacher_out,
    hard_targets=labels
)
```

## Loss-Task Zuordnung

| Task                   | Loss-Funktion                   | Module                |
| ---------------------- | ------------------------------- | --------------------- |
| Language Modeling      | `CrossEntropyLoss`              | GPT, DeepSeek         |
| Multi-Token Prediction | `MTPLoss`                       | DeepSeek-V3           |
| Object Detection       | `GIoULoss`, `FocalLoss`         | YOLO, EfficientDet    |
| CenterNet Detection    | `CenterNetLoss`                 | CenterNet             |
| Image Classification   | `CrossEntropyLoss`, `FocalLoss` | ResNet, EfficientNet  |
| Keypoint Detection     | `KeypointLoss`                  | Pose Estimation       |
| Segmentation           | `DiceLoss`, `IoULoss`           | Semantic Segmentation |
| Audio Processing       | `MelLoss`, `SNRLoss`            | Audio Models          |
| Knowledge Transfer     | `DistillationLoss`, `LwFLoss`   | Continual Learning    |

## Features

- ✅ Modulare Loss-Komponenten
- ✅ Kombination mehrerer Losses mit Gewichtung
- ✅ Unterstützung für Label Smoothing
- ✅ Class Imbalance Handling (Focal Loss)
- ✅ Gradient-freundliche Implementierungen
- ✅ Effiziente Batch-Verarbeitung

## Best Practices

### Loss-Kombination

```python
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.box_loss = GIoULoss()

    def forward(self, pred_cls, pred_box, target_cls, target_box):
        loss_cls = self.cls_loss(pred_cls, target_cls)
        loss_box = self.box_loss(pred_box, target_box)
        return loss_cls + 5.0 * loss_box  # Box-Loss höher gewichten
```

### Loss-Scheduling

```python
class AdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mtp_lambda = 0.3

    def forward(self, logits, mtp_logits, targets, epoch):
        # MTP-Gewicht über Epochen anpassen
        mtp_weight = self.mtp_lambda * min(1.0, epoch / 10)

        main_loss = F.cross_entropy(logits, targets)
        mtp_loss = self.compute_mtp_loss(mtp_logits, targets)

        return main_loss + mtp_weight * mtp_loss
```

## Weitere Informationen

- Hauptdokumentation: [docs/modules_overview.md](../../../docs/modules_overview.md)
- LLM Losses: [docs/llm_loss_head.md](../../../docs/llm_loss_head.md)
