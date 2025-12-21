Getting Started
===============

Willkommen bei APTT! Diese Anleitung hilft Ihnen beim Einstieg.

Installation
------------

Voraussetzungen
~~~~~~~~~~~~~~~

- Python >= 3.11
- PyTorch >= 2.6.0
- CUDA 12.4 (optional für GPU-Support)

Installation mit uv
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Repository klonen
   git clone https://github.com/afeldman/deepsuite.git
   cd deepsuite
   
   # Mit CPU-Support installieren
   uv install --extra cpu
   
   # Mit CUDA 12.4 Support installieren (Linux/Windows)
   uv install --extra cu124

Entwicklungsumgebung einrichten
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Für die Entwicklung:

.. code-block:: bash

   # Development Dependencies installieren
   uv install --extra dev
   
   # Dokumentation bauen
   uv install --extra doc
   cd docs
   make html

Grundkonzepte
-------------

BaseModule
~~~~~~~~~~

Das ``BaseModule`` ist die Grundlage für alle Modelle in APTT. Es erweitert 
``pytorch_lightning.LightningModule`` mit zusätzlichen Features:

- Automatisches Hyperparameter-Tuning mit Ray Tune
- MLflow-Integration für Experiment-Tracking
- Integrierte Metriken (Accuracy, F1, AUROC, etc.)
- Knowledge Distillation Support

Beispiel:

.. code-block:: python

   from deepsuite.lightning_base.module import BaseModule
   import torch.nn as nn
   
   class MyModel(BaseModule):
       def __init__(self, num_classes=10):
           super().__init__(
               num_classes=num_classes,
               loss_name="focal",
               metrics=["accuracy", "f1"]
           )
           self.model = nn.Sequential(
               nn.Linear(784, 128),
               nn.ReLU(),
               nn.Linear(128, num_classes)
           )
       
       def forward(self, x):
           return self.model(x)

BaseTrainer
~~~~~~~~~~~

Der ``BaseTrainer`` erweitert PyTorch Lightning's Trainer mit:

- Automatischer Batch-Size-Optimierung
- Export-Callbacks (ONNX, TensorRT, TorchScript)
- MLflow und TensorBoard Logging
- Early Stopping und Model Checkpointing

Beispiel:

.. code-block:: python

   from deepsuite.lightning_base.trainer import BaseTrainer
   
   trainer = BaseTrainer(
       log_dir="logs",
       mlflow_experiment="my_experiment",
       export_formats=["torchscript", "tensor_rt"],
       max_epochs=100
   )
   
   trainer.fit(model, datamodule=datamodule)

Konfigurationsmanagement
~~~~~~~~~~~~~~~~~~~~~~~~~

APTT bietet einen ``ConfigManager`` für einfache Konfigurationsverwaltung:

.. code-block:: python

   from deepsuite.config.config_manager import ConfigManager
   
   config_manager = ConfigManager(config_dir="configs")
   
   # Konfiguration speichern
   config = {
       "learning_rate": 0.001,
       "batch_size": 32,
       "num_epochs": 100
   }
   config_manager.save_yaml(config, "training_config")
   
   # Konfiguration laden
   config = config_manager.load_yaml("training_config")

Nächste Schritte
----------------

- :doc:`training` - Lernen Sie, wie man Modelle trainiert
- :doc:`detection` - Object Detection mit YOLO und CenterNet
- :doc:`continual_learning` - Continual Learning implementieren
