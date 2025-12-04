"""APTT - Advanced PyTorch Training Toolkit.

A PyTorch Lightning-based framework for deep learning with focus on:
- Object Detection (YOLO, CenterNet)
- Object Tracking
- Continual Learning
- Audio/Signal Processing
"""

__version__ = "0.1.0"

# Core Lightning Modules
from aptt.modules.yolo import Yolo
from aptt.modules.centernet import CenterNetModule
from aptt.modules.tracking import TrackingModule

# Base Classes
from aptt.lightning_base.module import BaseModule
from aptt.lightning_base.trainer import BaseTrainer
from aptt.lightning_base.continual_learning_manager import ContinualLearningManager

# Model Architectures
from aptt.model.detection.yolo import YOLO
from aptt.model.detection.centernet import CenterNetModel
from aptt.model.backend_adapter import BackboneAdapter

__all__ = [
    # Version
    "__version__",
    # Lightning Modules (end-to-end trainable)
    "Yolo",
    "CenterNetModule",
    "TrackingModule",
    # Base Classes
    "BaseModule",
    "BaseTrainer",
    "ContinualLearningManager",
    # Model Architectures
    "YOLO",
    "CenterNetModel",
    "BackboneAdapter",
]
