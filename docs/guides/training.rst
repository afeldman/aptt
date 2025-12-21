Training Guide
==============

Dieser Guide zeigt Ihnen, wie Sie Modelle mit APTT trainieren.

Einfaches Training
------------------

Modell definieren
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.lightning_base.module import BaseModule
   import torch.nn as nn
   
   class SimpleClassifier(BaseModule):
       def __init__(self, input_size=784, hidden_size=128, num_classes=10):
           super().__init__(
               num_classes=num_classes,
               loss_name="focal",
               metrics=["accuracy", "f1", "precision", "recall"]
           )
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(hidden_size, num_classes)
       
       def forward(self, x):
           x = x.view(x.size(0), -1)
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return x

DataModule erstellen
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.lightning_base.dataset.image_loader import ImageDataModule
   
   datamodule = ImageDataModule(
       train_dir="data/train",
       val_dir="data/val",
       batch_size=32,
       num_workers=4,
       image_size=(224, 224)
   )

Training starten
~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.lightning_base.trainer import BaseTrainer
   
   model = SimpleClassifier()
   
   trainer = BaseTrainer(
       log_dir="logs",
       mlflow_experiment="classification_experiment",
       max_epochs=50
   )
   
   trainer.fit(model, datamodule=datamodule)

Hyperparameter-Tuning
---------------------

Mit Ray Tune
~~~~~~~~~~~~

.. code-block:: python

   from ray import tune
   
   search_space = {
       "lr": tune.loguniform(1e-5, 1e-2),
       "batch_size": tune.choice([16, 32, 64, 128]),
       "hidden_size": tune.choice([64, 128, 256, 512])
   }
   
   model = SimpleClassifier(search_space=search_space)
   best_config = model.optimize_hyperparameters(
       datamodule=datamodule,
       num_samples=20,
       max_epochs=10
   )

Loss-Funktionen
---------------

APTT bietet eine Vielzahl von Loss-Funktionen:

.. code-block:: python

   from deepsuite.loss import get_loss
   
   # Focal Loss für unbalancierte Datasets
   focal_loss = get_loss("focal", alpha=0.25, gamma=2.0)
   
   # Binary Focal Loss
   binary_focal = get_loss("binaryfocal", alpha=0.25, gamma=2.0)
   
   # Varifocal Loss
   varifocal = get_loss("varfocal", alpha=0.75, gamma=2.0)
   
   # Bounding Box Loss
   bbox_loss = get_loss("bbox", iou_type="giou")

Verfügbare Losses:

- ``bbox`` - Bounding Box Loss (IoU, GIoU, DIoU, CIoU)
- ``focal`` - Focal Loss
- ``binaryfocal`` - Binary Focal Loss
- ``multiclassfocal`` - Multiclass Focal Loss
- ``centernet`` - CenterNet Loss
- ``detection`` - Detection Loss
- ``keypoint`` - Keypoint Loss
- ``segmentation`` - Segmentation Loss
- ``distill`` - Knowledge Distillation Loss
- ``lwf`` - Learning without Forgetting

Metriken
--------

Integrierte Metriken
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = BaseModule(
       num_classes=10,
       metrics=["accuracy", "f1", "auroc", "precision", "recall"]
   )

Verfügbare Metriken:

- ``accuracy`` - Accuracy
- ``f1`` - F1 Score
- ``auroc`` - Area Under ROC Curve
- ``precision`` - Precision
- ``recall`` - Recall

Callbacks
---------

Model Export Callbacks
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.callbacks.torchscript import TorchScriptExportCallback
   from deepsuite.callbacks.tensor_rt import TensorRTExportCallback
   
   trainer = BaseTrainer(
       export_formats=["torchscript", "tensor_rt"],
       model_output_dir="exported_models"
   )

Visualization Callbacks
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepsuite.callbacks.embedding_logger import EmbeddingLoggerCallback
   from deepsuite.callbacks.tsne_laplace_callback import TSNELaplaceCallback
   
   callbacks = [
       EmbeddingLoggerCallback(),
       TSNELaplaceCallback()
   ]
   
   trainer = BaseTrainer(callbacks=callbacks)

Early Stopping & Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
   
   early_stop = EarlyStopping(
       monitor="val/loss",
       patience=10,
       mode="min"
   )
   
   checkpoint = ModelCheckpoint(
       monitor="val/accuracy",
       mode="max",
       save_top_k=3,
       dirpath="checkpoints"
   )
   
   trainer = BaseTrainer(
       early_stopping=early_stop,
       model_checkpoint=checkpoint
   )

Mixed Precision Training
------------------------

.. code-block:: python

   trainer = BaseTrainer(
       precision="16-mixed"  # Oder "bf16-mixed" für BFloat16
   )

Multi-GPU Training
------------------

.. code-block:: python

   trainer = BaseTrainer(
       accelerator="gpu",
       devices=4,  # 4 GPUs verwenden
       strategy="ddp"  # Distributed Data Parallel
   )

Logging
-------

TensorBoard
~~~~~~~~~~~

.. code-block:: python

   trainer = BaseTrainer(
       log_dir="logs/tensorboard"
   )
   
   # TensorBoard starten:
   # tensorboard --logdir=logs/tensorboard

MLflow
~~~~~~

.. code-block:: python

   trainer = BaseTrainer(
       mlflow_experiment="my_experiment"
   )

Beide Logger werden automatisch aktiviert und loggen:

- Loss-Werte (Training & Validation)
- Metriken (Accuracy, F1, etc.)
- Lernrate
- Model-Checkpoints
- Hyperparameter

Best Practices
--------------

1. **Batch Size optimieren**: Nutzen Sie ``auto_batch_size=True``
2. **Gradient Clipping**: ``trainer = BaseTrainer(gradient_clip_val=1.0)``
3. **Learning Rate Finder**: Nutzen Sie den Tuner
4. **Checkpointing**: Speichern Sie regelmäßig
5. **Early Stopping**: Vermeiden Sie Overfitting
6. **Mixed Precision**: Beschleunigt Training auf modernen GPUs
