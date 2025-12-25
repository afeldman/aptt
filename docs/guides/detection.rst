Object Detection
================

APTT bietet umfassende Unterstützung für Object Detection mit YOLO und CenterNet.

YOLO
----

YOLO (You Only Look Once) ist ein beliebtes Echtzeit-Object-Detection-Framework.

Modell erstellen
~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.modules.yolo import Yolo
   from deepsuite.model.feature.efficientnet import EfficientNetBackbone

   # Backbone erstellen
   backbone = EfficientNetBackbone(
       resolution_coefficient=1.0,
       width_coefficient=1.0,
       depth_coefficient=1.0,
       version="b0"
   )

   # YOLO Modul erstellen
   model = Yolo(
       backbone=backbone,
       num_classes=80,  # COCO hat 80 Klassen
       reg_max=16,
       use_rotated_loss=False
   )

Mit ResNet Backbone
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.model.feature.resnet import ResNetBackbone

   backbone = ResNetBackbone(
       resnet_variant=[[64, 128, 256, 512], [3, 4, 6, 3], 4, True],
       in_channels=3,
       stage_indices=(3, 4, 5)
   )

   model = Yolo(
       backbone=backbone,
       num_classes=20  # VOC hat 20 Klassen
   )

Training
~~~~~~~~

.. code-block:: python

   from deepsuite.lightning_base.trainer import BaseTrainer
   from pytorch_lightning import LightningDataModule

   trainer = BaseTrainer(
       max_epochs=300,
       mlflow_experiment="yolo_training",
       export_formats=["torchscript"]
   )

   trainer.fit(model, datamodule=detection_datamodule)

CenterNet
---------

CenterNet detektiert Objekte als Punkte (Zentren) anstelle von Bounding Boxes.

Modell erstellen
~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.modules.centernet import CenterNetModule
   from deepsuite.model.feature.resnet import ResNetBackbone

   # Backbone
   backbone = ResNetBackbone(
       resnet_variant=[[64, 128, 256, 512], [3, 4, 6, 3], 4, True],
       in_channels=3,
       stage_indices=(2, 3, 4)
   )

   # CenterNet Modul
   model = CenterNetModule(
       backbone=backbone,
       in_channels_list=[256, 512, 1024],
       num_classes=80,
       lr=1e-3,
       decoder_topk=100
   )

Training
~~~~~~~~

.. code-block:: python

   trainer = BaseTrainer(
       max_epochs=140,
       mlflow_experiment="centernet_training"
   )

   trainer.fit(model, datamodule=detection_datamodule)

Detection Losses
----------------

Bounding Box Loss
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.loss.bbox import BboxLoss

   # Standard IoU
   loss = BboxLoss(iou_type="iou")

   # GIoU (Generalized IoU)
   loss = BboxLoss(iou_type="giou")

   # DIoU (Distance IoU)
   loss = BboxLoss(iou_type="diou")

   # CIoU (Complete IoU)
   loss = BboxLoss(iou_type="ciou")

Rotated Bounding Box Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~

Für rotierte Bounding Boxes:

.. code-block:: python

   from deepsuite.loss.bbox import RotatedBboxLoss

   loss = RotatedBboxLoss()

   model = Yolo(
       backbone=backbone,
       use_rotated_loss=True
   )

CenterNet Loss
~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.loss.centernet import CenterNetLoss

   loss = CenterNetLoss(
       heatmap_loss_weight=1.0,
       offset_loss_weight=1.0,
       size_loss_weight=0.1
   )

Detection Metrics
-----------------

mAP (Mean Average Precision)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.metric.map import evaluate_map

   map_score = evaluate_map(
       pred_boxes=predictions,
       gt_boxes=ground_truth,
       iou_thresh=0.5
   )

IoU Metrics
~~~~~~~~~~~

.. code-block:: python

   from deepsuite.metric.bbox_iou import bbox_iou

   # Standard IoU
   iou = bbox_iou(boxes1, boxes2, xywh=True, iou_type="iou")

   # GIoU
   giou = bbox_iou(boxes1, boxes2, xywh=True, iou_type="giou")

   # Rotated IoU
   from deepsuite.metric.bbox_iou import rotated_bbox_iou
   rot_iou = rotated_bbox_iou(rotated_boxes1, rotated_boxes2)

Detection Metrics Module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.metric.detection import DetectionMetrics

   metrics = DetectionMetrics(num_classes=80)

   # Im Validation Step
   def validation_step(self, batch, batch_idx):
       x, targets = batch
       predictions = self(x)

       metrics.update(predictions, targets)

   def on_validation_epoch_end(self):
       results = metrics.compute()
       self.log("val/map", results["map"])

Inference
---------

YOLO Inference
~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from PIL import Image
   import torchvision.transforms as T

   # Modell laden
   model = Yolo.load_from_checkpoint("checkpoint.ckpt")
   model.eval()

   # Bild vorbereiten
   image = Image.open("image.jpg")
   transform = T.Compose([
       T.Resize((640, 640)),
       T.ToTensor(),
   ])
   x = transform(image).unsqueeze(0)

   # Prediction
   with torch.no_grad():
       outputs = model(x)

   boxes = outputs["bbox"]
   classes = outputs["class"]

CenterNet Inference
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Modell laden
   model = CenterNetModule.load_from_checkpoint("checkpoint.ckpt")
   model.eval()

   # Prediction
   with torch.no_grad():
       outputs = model(x)

   # Decode predictions
   detections = model.decoder(outputs, img_size=(640, 640))

Nachbearbeitung
~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.utils.bbox import xywh2xyxy, xyxy2xywh

   # Convert zwischen Formaten
   xyxy_boxes = xywh2xyxy(xywh_boxes)
   xywh_boxes = xyxy2xywh(xyxy_boxes)

   # NMS (Non-Maximum Suppression)
   from torchvision.ops import nms

   keep = nms(boxes, scores, iou_threshold=0.5)
   filtered_boxes = boxes[keep]
   filtered_scores = scores[keep]

Custom Detection Head
---------------------

.. code-block:: python

   from deepsuite.heads.box import BBoxHead
   import torch.nn as nn

   class CustomDetectionHead(nn.Module):
       def __init__(self, in_channels, num_classes):
           super().__init__()
           self.bbox_head = BBoxHead(in_channels)
           self.class_head = nn.Conv2d(
               in_channels,
               num_classes,
               kernel_size=1
           )

       def forward(self, x):
           bbox = self.bbox_head(x)
           classes = self.class_head(x)
           return {"bbox": bbox, "class": classes}

Best Practices
--------------

1. **Data Augmentation**: Nutzen Sie Augmentationen
2. **Learning Rate Scheduling**: Cosine Annealing oder Step LR
3. **Warm-up**: Starten Sie mit niedriger Learning Rate
4. **Anchor-Free**: CenterNet ist oft einfacher zu tunen
5. **Multi-Scale Training**: Verbessert Robustheit
6. **Mosaic Augmentation**: Besonders effektiv für YOLO
7. **Model Pruning**: Für Deployment optimieren
