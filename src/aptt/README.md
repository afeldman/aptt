# APTT Source Code

Dies ist das Hauptverzeichnis für den APTT (Antons PyTorch Tools) Source Code.

## Modulübersicht

Jedes Unterverzeichnis enthält eine eigene README.md mit detaillierter Dokumentation.

### 🧠 Language Models & NLP

- **[layers/](layers/)** - Neuronale Netzwerk-Layer

  - Attention Mechanisms (MLA, RoPE, KV-Compression)
  - Mixture-of-Experts (MoE)
  - Specialized Layers (Complex, Hermite, Laguerre)

- **[modules/](modules/)** - PyTorch Lightning Module

  - GPT-2/GPT-3
  - DeepSeek-V3 mit MLA und MoE
  - YOLO, CenterNet

- **[heads/](heads/)** - Ausgabe-Köpfe

  - Language Model Head
  - Multi-Token Prediction Head
  - Classification, Detection, Heatmap Heads

- **[loss/](loss/)** - Verlustfunktionen
  - Language Modeling Losses
  - Multi-Token Prediction Loss
  - Detection Losses (Focal, GIoU, DFL)
  - Knowledge Distillation

### 👁️ Computer Vision

- **[model/](model/)** - Modell-Architekturen

  - Object Detection (YOLO, CenterNet, EfficientDet)
  - Feature Extraction (ResNet, EfficientNet, DarkNet, FPN)
  - Tracking

- **[tracker/](tracker/)** - Multi-Object Tracking

  - SORT, DeepSORT, ByteTrack
  - Re-Identification
  - Tracking Pipeline

- **[metric/](metric/)** - Evaluations-Metriken
  - Mean Average Precision (mAP)
  - Detection Metrics
  - Confusion Matrix

### 🎵 Audio Processing

- **[model/beamforming/](model/beamforming/)** - Audio Beamforming
- Siehe auch: `model/complex.py`, `model/doa.py`, `model/rnn.py`

### ⚙️ Training & Utilities

- **[lightning_base/](lightning_base/)** - PyTorch Lightning Basis

  - Base Lightning Module
  - Dataset Loaders (Text, Image, Audio)
  - Continual Learning Manager

- **[callbacks/](callbacks/)** - Training Callbacks

  - TorchScript, TensorRT Export
  - Embedding Logger
  - t-SNE Visualization

- **[utils/](utils/)** - Utility-Funktionen

  - Bounding Box Operations
  - Image Processing
  - Tensor Utilities
  - Device Management

- **[viz/](viz/)** - Visualisierung
  - Embedding Visualization (t-SNE, UMAP, PCA)

### 🔧 Specialized

- **[config/](config/)** - Konfigurationsmanagement
- **[svm/](svm/)** - Support Vector Machines
- **[conv/](conv/)** - Konventionelle Convolutional Networks

## Schnellzugriff

### Wichtigste Dateien

```
src/aptt/
├── modules/
│   ├── gpt.py              # GPT-2/GPT-3 Implementation
│   ├── deepseek.py         # DeepSeek-V3 Implementation
│   ├── yolo.py             # YOLO Object Detection
│   └── centernet.py        # CenterNet Detection
│
├── layers/
│   ├── attention/          # Attention Mechanisms
│   │   ├── mla.py          # Multi-Head Latent Attention
│   │   └── rope.py         # Rotary Position Embeddings
│   └── moe.py              # Mixture-of-Experts
│
├── lightning_base/dataset/
│   ├── text_loader.py      # Text Dataset für LLMs
│   ├── image_loader.py     # Image Dataset
│   └── audio_loader.py     # Audio Dataset
│
└── loss/
    ├── classification.py   # Cross-Entropy, Focal Loss
    ├── mtp_loss.py         # Multi-Token Prediction Loss
    └── detection.py        # Detection Losses
```

## Verwendung

### Import-Beispiele

```python
# Language Models
from aptt.modules import GPTModule, DeepSeekModule
from aptt.layers.attention import MultiHeadLatentAttention
from aptt.layers import DeepSeekMoE

# Object Detection
from aptt.modules import YOLOModule, CenterNetModule
from aptt.model.detection import YOLO, CenterNet

# Datasets
from aptt.lightning_base.dataset import TextDataLoader, ImageDataLoader

# Losses
from aptt.loss import CrossEntropyLoss, MTPLoss, FocalLoss, GIoULoss

# Metrics
from aptt.metric import MeanAveragePrecision, DetectionMetrics

# Tracking
from aptt.tracker import ObjectTracker, DeepSORTTracker

# Utils
from aptt.utils import bbox, image, device

# Visualization
from aptt.viz import EmbeddingVisualizer
```

## Dokumentation

### READMEs

Jedes Modul hat eine eigene README.md:

- [callbacks/README.md](callbacks/README.md) - Training Callbacks
- [heads/README.md](heads/README.md) - Output Heads
- [layers/README.md](layers/README.md) - Neural Network Layers
- [lightning_base/README.md](lightning_base/README.md) - Lightning Base Components
- [loss/README.md](loss/README.md) - Loss Functions
- [metric/README.md](metric/README.md) - Evaluation Metrics
- [model/README.md](model/README.md) - Model Architectures
- [modules/README.md](modules/README.md) - Lightning Modules
- [tracker/README.md](tracker/README.md) - Object Tracking
- [utils/README.md](utils/README.md) - Utility Functions
- [viz/README.md](viz/README.md) - Visualization Tools

### Vollständige Dokumentation

Siehe [docs/modules_overview.md](../../docs/modules_overview.md) für eine komplette Übersicht aller Module.

### Spezifische Themen

- [docs/llm_modules.md](../../docs/llm_modules.md) - Language Models (GPT, DeepSeek-V3)
- [docs/llm_loss_head.md](../../docs/llm_loss_head.md) - LLM Losses & Heads
- [docs/moe.md](../../docs/moe.md) - Mixture-of-Experts
- [docs/text_dataset.md](../../docs/text_dataset.md) - Text Data Loading

## Entwicklung

### Code-Stil

```bash
# Format
ruff format src/aptt

# Lint
ruff check src/aptt

# Type Checking
mypy src/aptt
```

### Tests

```bash
# Alle Tests
pytest tests/

# Spezifisches Modul
pytest tests/test_tensor_rt_export_callback.py
```

## Architektur-Prinzipien

### 1. Modularität

Jede Komponente ist unabhängig verwendbar:

```python
# Layer alleine
from aptt.layers import DeepSeekMoE
moe = DeepSeekMoE(d_model=2048, ...)

# In eigenem Model
class MyModel(nn.Module):
    def __init__(self):
        self.moe = DeepSeekMoE(...)
```

### 2. Lightning-Integration

Alle Haupt-Modelle sind Lightning Modules:

```python
from aptt.modules import DeepSeekModule
import pytorch_lightning as pl

model = DeepSeekModule(...)
trainer = pl.Trainer(...)
trainer.fit(model, datamodule)
```

### 3. Komposierbarkeit

Komponenten können frei kombiniert werden:

```python
from aptt.heads import LMHead
from aptt.loss import CrossEntropyLoss
from aptt.metric import Perplexity

class CustomModule(pl.LightningModule):
    def __init__(self):
        self.backbone = ...
        self.head = LMHead(...)
        self.loss_fn = CrossEntropyLoss(...)
        self.metric = Perplexity()
```

## Weitere Informationen

- **Hauptprojekt**: [README.md](../../README.md)
- **Dokumentation**: [docs/](../../docs/)
- **Beispiele**: [examples/](../../examples/)
- **Tests**: [tests/](../../tests/)

---

**Version**: 0.2.0 | **Lizenz**: MIT
