APTT – Antons PyTorch Tools
============================

**APTT** (Antons PyTorch Tools) ist ein modulares, erweiterbares Deep Learning Framework, 
das auf PyTorch Lightning basiert und Training, Evaluierung und Experimente vereinfacht.

Features
--------

- ✅ Breite Palette an unterstützten Modelltypen (YOLO, ResNet, RNNs, WaveNet, etc.)
- 🧩 Plugbare Callbacks (TorchScript-Export, TensorRT-Optimierung, t-SNE-Visualisierung, etc.)
- 🧠 Integriertes Continual Learning und Knowledge Distillation
- ⚙️ Modularer Aufbau (Heads, Losses, Layers, Metrics, Callbacks, etc.)
- 📊 Embedding-Visualisierung und Analyse-Tools
- 🗂️ Flexible Dataset-Loader für Audio- und Bildaufgaben
- 🧪 Unit-Tests und vollständige Dokumentation mit Sphinx

Installation
------------

.. code-block:: bash

   # Repository klonen
   git clone https://github.com/afeldman/aptt.git
   cd aptt

   # (Optional) Virtuelle Umgebung erstellen
   python -m venv venv
   source venv/bin/activate

   # Abhängigkeiten installieren
   uv install .

Quick Start
-----------

.. code-block:: python

   from aptt.lightning_base.trainer import BaseTrainer
   from aptt.lightning_base.module import BaseModule
   
   # Trainer erstellen
   trainer = BaseTrainer(
       log_dir="logs",
       mlflow_experiment="my_experiment"
   )
   
   # Modell trainieren
   trainer.fit(model, datamodule=datamodule)

.. toctree::
   :maxdepth: 2
   :caption: Dokumentation:

   guides/getting_started
   guides/training
   guides/detection
   guides/continual_learning
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Beispiele:

   examples/yolo
   examples/centernet
   examples/tracking

Indizes und Tabellen
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

