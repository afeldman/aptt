DeepSuite Documentation
========================

**DeepSuite** is a comprehensive deep learning framework built on PyTorch Lightning, featuring state-of-the-art implementations for:

* **Object Detection**: YOLO, CenterNet
* **Language Models**: GPT, DeepSeek-V3 with MLA/MoE
* **Specialized Models**: Audio/vision models, autoencoders
* **Object Tracking**: Advanced tracking algorithms
* **Continual Learning**: Adaptive learning strategies

.. image:: logo.png
   :align: center
   :width: 300px
   :alt: DeepSuite Logo

Features
--------

* 🚀 **Modern Architecture**: Built on PyTorch Lightning for scalable training
* 📦 **Modular Design**: Pluggable heads, backbones, and loss functions
* 🎯 **Production Ready**: Comprehensive testing with ruff and mypy
* 📚 **Well Documented**: Complete Google-style docstrings
* 🔧 **Flexible**: Support for various detection and classification tasks

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install with CUDA 12.4 support
   pip install deepsuite[cu124]

   # Or CPU-only version
   pip install deepsuite[cpu]

   # Development installation
   pip install deepsuite[dev,doc]

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepsuite import YOLO, BaseTrainer
   from deepsuite.heads import ClassificationHead
   
   # Create a YOLO model
   model = YOLO(
       num_classes=80,
       backbone="darknet53",
       pretrained=True
   )
   
   # Initialize trainer
   trainer = BaseTrainer(
       max_epochs=100,
       accelerator="gpu",
       devices=1
   )
   
   # Train the model
   trainer.fit(model, datamodule)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/getting_started
   guides/training
   guides/detection
   guides/continual_learning

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/yolo
   examples/centernet
   examples/tracking

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
