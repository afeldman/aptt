CenterNet Beispiel
==================

Vollständiges Beispiel für Object Detection mit CenterNet.

Training Setup
--------------

.. code-block:: python

   import torch
   from deepsuite.modules.centernet import CenterNetModule
   from deepsuite.model.feature.resnet import ResNetBackbone
   from deepsuite.lightning_base.trainer import BaseTrainer
   from deepsuite.lightning_base.dataset.image_loader import ImageDataModule
   
   # 1. ResNet Backbone erstellen
   backbone = ResNetBackbone(
       resnet_variant=[[64, 128, 256, 512], [3, 4, 6, 3], 4, True],  # ResNet-50
       in_channels=3,
       stage_indices=(2, 3, 4)  # Welche Features extrahiert werden
   )
   
   # 2. CenterNet Modell
   model = CenterNetModule(
       backbone=backbone,
       in_channels_list=[256, 512, 1024],  # Feature-Dimensionen
       num_classes=80,
       lr=1e-3,
       decoder_topk=100  # Top-K Detektionen
   )
   
   # 3. DataModule
   datamodule = ImageDataModule(
       train_dir="data/coco/train2017",
       val_dir="data/coco/val2017",
       batch_size=24,
       num_workers=8,
       image_size=(512, 512)
   )
   
   # 4. Trainer
   trainer = BaseTrainer(
       log_dir="logs/centernet",
       mlflow_experiment="centernet_coco",
       max_epochs=140,
       precision="16-mixed"
   )
   
   # 5. Training
   trainer.fit(model, datamodule=datamodule)

Inference
---------

.. code-block:: python

   import torch
   from PIL import Image
   import torchvision.transforms as T
   import numpy as np
   
   # Modell laden
   model = CenterNetModule.load_from_checkpoint("best.ckpt")
   model.eval()
   model.cuda()
   
   # Bild vorbereiten
   image = Image.open("test.jpg")
   transform = T.Compose([
       T.Resize((512, 512)),
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
   ])
   
   x = transform(image).unsqueeze(0).cuda()
   
   # Prediction
   with torch.no_grad():
       outputs = model(x)
       detections = model.decoder(outputs, img_size=(512, 512))
   
   # Detections extrahieren
   for det in detections:
       boxes = det["boxes"]  # [N, 4]
       scores = det["scores"]  # [N]
       classes = det["classes"]  # [N]
       
       # Filtern nach Confidence
       mask = scores > 0.3
       boxes = boxes[mask]
       scores = scores[mask]
       classes = classes[mask]
       
       print(f"Found {len(boxes)} detections")
       for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
           print(f"  {i+1}. Class {cls}: {score:.3f} at {box.tolist()}")

Visualisierung
--------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.patches as patches
   
   def visualize_detections(image_path, model, threshold=0.3):
       # Bild laden
       image = Image.open(image_path)
       original_size = image.size
       
       # Preprocessing
       transform = T.Compose([
           T.Resize((512, 512)),
           T.ToTensor(),
           T.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
       ])
       
       x = transform(image).unsqueeze(0).cuda()
       
       # Prediction
       model.eval()
       with torch.no_grad():
           outputs = model(x)
           detections = model.decoder(outputs, img_size=(512, 512))
       
       # Plot
       fig, ax = plt.subplots(1, figsize=(12, 9))
       ax.imshow(image)
       
       det = detections[0]
       boxes = det["boxes"].cpu()
       scores = det["scores"].cpu()
       classes = det["classes"].cpu()
       
       # Filtern
       mask = scores > threshold
       boxes = boxes[mask]
       scores = scores[mask]
       classes = classes[mask]
       
       # Zeichnen
       for box, score, cls in zip(boxes, scores, classes):
           x1, y1, x2, y2 = box.tolist()
           
           # Zurückskalieren
           x1 = x1 * original_size[0] / 512
           x2 = x2 * original_size[0] / 512
           y1 = y1 * original_size[1] / 512
           y2 = y2 * original_size[1] / 512
           
           width = x2 - x1
           height = y2 - y1
           
           rect = patches.Rectangle(
               (x1, y1), width, height,
               linewidth=2, edgecolor='lime', facecolor='none'
           )
           ax.add_patch(rect)
           
           # Center point
           cx = (x1 + x2) / 2
           cy = (y1 + y2) / 2
           ax.plot(cx, cy, 'ro', markersize=8)
           
           # Label
           label = f"C{int(cls)}: {score:.2f}"
           ax.text(x1, y1-10, label,
                  bbox=dict(facecolor='yellow', alpha=0.7),
                  fontsize=10, color='black')
       
       plt.axis('off')
       plt.tight_layout()
       plt.savefig("centernet_result.jpg", dpi=150, bbox_inches='tight')
       plt.show()
   
   # Verwendung
   visualize_detections("test.jpg", model, threshold=0.3)

Heatmap Visualisierung
-----------------------

.. code-block:: python

   def visualize_heatmaps(image_path, model):
       image = Image.open(image_path)
       transform = T.Compose([
           T.Resize((512, 512)),
           T.ToTensor(),
           T.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
       ])
       
       x = transform(image).unsqueeze(0).cuda()
       
       model.eval()
       with torch.no_grad():
           outputs = model(x)
       
       # Heatmaps extrahieren
       heatmaps = []
       for output in outputs:
           if "heatmap" in output:
               hm = output["heatmap"]
               heatmaps.append(hm)
       
       # Visualisierung
       num_scales = len(heatmaps)
       fig, axes = plt.subplots(1, num_scales + 1, figsize=(20, 5))
       
       # Original
       axes[0].imshow(image)
       axes[0].set_title("Original")
       axes[0].axis('off')
       
       # Heatmaps für jede Skala
       for i, hm in enumerate(heatmaps):
           # Summiere über alle Klassen
           hm_sum = hm[0].sum(dim=0).cpu().numpy()
           
           axes[i+1].imshow(image, alpha=0.5)
           im = axes[i+1].imshow(hm_sum, alpha=0.5, cmap='jet')
           axes[i+1].set_title(f"Scale {i+1}")
           axes[i+1].axis('off')
           plt.colorbar(im, ax=axes[i+1])
       
       plt.tight_layout()
       plt.savefig("centernet_heatmaps.jpg", dpi=150)
       plt.show()
   
   visualize_heatmaps("test.jpg", model)

Custom CenterNet Head
----------------------

.. code-block:: python

   from deepsuite.heads.centernet import CenterNetHead
   import torch.nn as nn
   
   class CustomCenterNetHead(nn.Module):
       def __init__(self, in_channels, num_classes):
           super().__init__()
           
           # Heatmap für Objektzentren
           self.heatmap_head = nn.Sequential(
               nn.Conv2d(in_channels, 256, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(256, num_classes, 1)
           )
           
           # Größen-Regression
           self.size_head = nn.Sequential(
               nn.Conv2d(in_channels, 256, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(256, 2, 1)
           )
           
           # Offset für Sub-Pixel Genauigkeit
           self.offset_head = nn.Sequential(
               nn.Conv2d(in_channels, 256, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(256, 2, 1)
           )
       
       def forward(self, x):
           heatmap = torch.sigmoid(self.heatmap_head(x))
           size = self.size_head(x)
           offset = self.offset_head(x)
           
           return {
               "heatmap": heatmap,
               "size": size,
               "offset": offset
           }

Evaluation
----------

.. code-block:: python

   from deepsuite.metric.detection import DetectionMetrics
   from torch.utils.data import DataLoader
   
   def evaluate_centernet(model, dataloader, num_classes=80):
       metrics = DetectionMetrics(num_classes=num_classes)
       
       model.eval()
       with torch.no_grad():
           for batch in dataloader:
               images, targets = batch
               images = images.cuda()
               
               outputs = model(images)
               detections = model.decoder(outputs, img_size=(512, 512))
               
               metrics.update(detections, targets)
       
       results = metrics.compute()
       
       print(f"mAP@0.5: {results['map_50']:.4f}")
       print(f"mAP@0.5:0.95: {results['map']:.4f}")
       print(f"Recall: {results['recall']:.4f}")
       print(f"Precision: {results['precision']:.4f}")
       
       return results
   
   # Verwendung
   results = evaluate_centernet(model, val_loader)

Multi-Scale Testing
-------------------

.. code-block:: python

   def multi_scale_inference(model, image, scales=[0.5, 1.0, 1.5]):
       all_detections = []
       
       for scale in scales:
           size = int(512 * scale)
           transform = T.Compose([
               T.Resize((size, size)),
               T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
           ])
           
           x = transform(image).unsqueeze(0).cuda()
           
           with torch.no_grad():
               outputs = model(x)
               detections = model.decoder(outputs, img_size=(size, size))
           
           all_detections.extend(detections)
       
       # NMS über alle Skalen
       from torchvision.ops import nms
       
       all_boxes = torch.cat([d["boxes"] for d in all_detections])
       all_scores = torch.cat([d["scores"] for d in all_detections])
       all_classes = torch.cat([d["classes"] for d in all_detections])
       
       keep = nms(all_boxes, all_scores, iou_threshold=0.5)
       
       return {
           "boxes": all_boxes[keep],
           "scores": all_scores[keep],
           "classes": all_classes[keep]
       }
