YOLO Beispiel
=============

Vollständiges Beispiel für Object Detection mit YOLO.

Komplettes Training
-------------------

.. code-block:: python

   import torch
   from pathlib import Path
   from deepsuite.modules.yolo import Yolo
   from deepsuite.model.feature.efficientnet import EfficientNetBackbone
   from deepsuite.lightning_base.trainer import BaseTrainer
   from deepsuite.lightning_base.dataset.image_loader import ImageDataModule
   
   # 1. Backbone erstellen
   backbone = EfficientNetBackbone(
       resolution_coefficient=1.0,
       width_coefficient=1.0,
       depth_coefficient=1.0,
       version="b0"
   )
   
   # 2. YOLO Modell erstellen
   model = Yolo(
       backbone=backbone,
       num_classes=80,  # COCO Dataset
       reg_max=16,
       use_rotated_loss=False,
       metrics=["accuracy", "precision", "recall"]
   )
   
   # 3. DataModule konfigurieren
   datamodule = ImageDataModule(
       train_dir="data/coco/train2017",
       val_dir="data/coco/val2017",
       batch_size=16,
       num_workers=8,
       image_size=(640, 640),
       augmentation=True
   )
   
   # 4. Trainer konfigurieren
   trainer = BaseTrainer(
       log_dir="logs/yolo_training",
       mlflow_experiment="yolo_coco",
       max_epochs=300,
       export_formats=["torchscript"],
       auto_batch_size=True,
       precision="16-mixed"  # Mixed Precision Training
   )
   
   # 5. Training starten
   trainer.fit(model, datamodule=datamodule)
   
   # 6. Modell exportieren
   # TorchScript wurde automatisch exportiert

Inference
---------

.. code-block:: python

   import torch
   from PIL import Image
   import torchvision.transforms as T
   import matplotlib.pyplot as plt
   import matplotlib.patches as patches
   
   # Modell laden
   model = Yolo.load_from_checkpoint("checkpoints/best.ckpt")
   model.eval()
   model.cuda()
   
   # Bild laden und preprocessen
   image = Image.open("test_image.jpg")
   original_size = image.size
   
   transform = T.Compose([
       T.Resize((640, 640)),
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
   ])
   
   x = transform(image).unsqueeze(0).cuda()
   
   # Prediction
   with torch.no_grad():
       outputs = model(x)
   
   # Ergebnisse verarbeiten
   boxes = outputs["bbox"].cpu()  # [N, 4]
   classes = outputs["class"].cpu()  # [N, num_classes]
   scores, class_ids = classes.max(dim=1)
   
   # Confidence threshold
   threshold = 0.5
   mask = scores > threshold
   
   filtered_boxes = boxes[mask]
   filtered_scores = scores[mask]
   filtered_classes = class_ids[mask]
   
   # Visualisierung
   fig, ax = plt.subplots(1, figsize=(12, 9))
   ax.imshow(image)
   
   for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes):
       x1, y1, x2, y2 = box.tolist()
       
       # Zurück zur Originalgröße skalieren
       x1 = x1 * original_size[0] / 640
       x2 = x2 * original_size[0] / 640
       y1 = y1 * original_size[1] / 640
       y2 = y2 * original_size[1] / 640
       
       width = x2 - x1
       height = y2 - y1
       
       rect = patches.Rectangle(
           (x1, y1), width, height,
           linewidth=2, edgecolor='r', facecolor='none'
       )
       ax.add_patch(rect)
       
       # Label
       label = f"Class {cls.item()}: {score:.2f}"
       ax.text(x1, y1-10, label, 
               bbox=dict(facecolor='yellow', alpha=0.5),
               fontsize=9, color='black')
   
   plt.axis('off')
   plt.tight_layout()
   plt.savefig("detection_result.jpg", dpi=150, bbox_inches='tight')
   plt.show()

Batch Inference
---------------

.. code-block:: python

   from torch.utils.data import DataLoader
   from tqdm import tqdm
   
   # Dataset für Inference
   class InferenceDataset(torch.utils.data.Dataset):
       def __init__(self, image_paths, transform):
           self.image_paths = image_paths
           self.transform = transform
       
       def __len__(self):
           return len(self.image_paths)
       
       def __getitem__(self, idx):
           image = Image.open(self.image_paths[idx])
           return self.transform(image), self.image_paths[idx]
   
   # DataLoader erstellen
   dataset = InferenceDataset(image_paths, transform)
   loader = DataLoader(dataset, batch_size=8, num_workers=4)
   
   model.eval()
   results = []
   
   with torch.no_grad():
       for batch, paths in tqdm(loader):
           batch = batch.cuda()
           outputs = model(batch)
           
           for i, path in enumerate(paths):
               boxes = outputs["bbox"][i].cpu()
               classes = outputs["class"][i].cpu()
               
               results.append({
                   "path": path,
                   "boxes": boxes,
                   "classes": classes
               })
   
   # Ergebnisse speichern
   import json
   with open("detection_results.json", "w") as f:
       json.dump(results, f, indent=2)

Video Inference
---------------

.. code-block:: python

   import cv2
   
   # Video öffnen
   cap = cv2.VideoCapture("input_video.mp4")
   
   # Output Video Writer
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   fps = int(cap.get(cv2.CAP_PROP_FPS))
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))
   
   model.eval()
   
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       # Frame zu Tensor
       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       image_pil = Image.fromarray(frame_rgb)
       x = transform(image_pil).unsqueeze(0).cuda()
       
       # Prediction
       with torch.no_grad():
           outputs = model(x)
       
       # Detektionen zeichnen
       boxes = outputs["bbox"][0].cpu()
       classes = outputs["class"][0].cpu()
       scores, class_ids = classes.max(dim=0)
       
       for box, score, cls in zip(boxes, scores, class_ids):
           if score > 0.5:
               x1, y1, x2, y2 = box.tolist()
               
               # Skalierung
               x1 = int(x1 * width / 640)
               x2 = int(x2 * width / 640)
               y1 = int(y1 * height / 640)
               y2 = int(y2 * height / 640)
               
               # Box zeichnen
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               
               # Label
               label = f"Class {cls.item()}: {score:.2f}"
               cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       
       out.write(frame)
   
   cap.release()
   out.release()

Export und Deployment
---------------------

ONNX Export
~~~~~~~~~~~

.. code-block:: python

   import torch.onnx
   
   model.eval()
   dummy_input = torch.randn(1, 3, 640, 640).cuda()
   
   torch.onnx.export(
       model,
       dummy_input,
       "yolo_model.onnx",
       export_params=True,
       opset_version=11,
       do_constant_folding=True,
       input_names=['input'],
       output_names=['bbox', 'class'],
       dynamic_axes={
           'input': {0: 'batch_size'},
           'bbox': {0: 'batch_size'},
           'class': {0: 'batch_size'}
       }
   )

TensorRT Optimierung
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # TensorRT wurde automatisch exportiert während des Trainings
   # wenn export_formats=["tensor_rt"] gesetzt war
   
   # Oder manuell:
   import torch_tensorrt
   
   trt_model = torch_tensorrt.ts.compile(
       model,
       inputs=[torch_tensorrt.Input((1, 3, 640, 640))],
       enabled_precisions={torch.float, torch.half},
       workspace_size=1 << 30
   )
   
   torch.jit.save(trt_model, "yolo_trt.ts")

Mobile Deployment (TorchScript)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # TorchScript wurde automatisch exportiert
   scripted_model = torch.jit.load("yolo_torchscript.pt")
   
   # Für Mobile optimieren
   optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(
       scripted_model
   )
   
   optimized_model._save_for_lite_interpreter("yolo_mobile.ptl")

Hyperparameter Tuning
---------------------

.. code-block:: python

   from ray import tune
   
   search_space = {
       "lr": tune.loguniform(1e-5, 1e-3),
       "batch_size": tune.choice([8, 16, 32]),
       "reg_max": tune.choice([8, 16, 32]),
   }
   
   model = Yolo(
       backbone=backbone,
       num_classes=80,
       search_space=search_space
   )
   
   best_config = model.optimize_hyperparameters(
       datamodule=datamodule,
       num_samples=20,
       max_epochs=50
   )
   
   print(f"Best config: {best_config}")
