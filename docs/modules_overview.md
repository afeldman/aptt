# APTT Modules Documentation

Übersicht über alle implementierten Module und deren Dokumentation.

## 🧠 Language Models & NLP

### Transformer Language Models

#### DeepSeek-V3

**Dokumentation:** [docs/llm_modules.md](llm_modules.md)

State-of-the-art Language Model mit fortgeschrittenen Techniken:

- **Multi-Head Latent Attention (MLA)**: Effiziente Attention mit KV-Compression
- **Mixture-of-Experts (MoE)**: Sparse Expert Activation für hohe Kapazität
- **Multi-Token Prediction (MTP)**: Simultane Prediction mehrerer zukünftiger Tokens
- **Rotary Position Embeddings (RoPE)**: Relative Positionskodierung

**Verwendung:**

```python
from aptt.modules.deepseek import DeepSeekModule

model = DeepSeekModule(
    vocab_size=50000,
    d_model=2048,
    n_layers=24,
    use_moe=True,
    use_mtp=True,
)
```

**Features:**

- 37B aktive Parameter (685B total für full scale)
- Auxiliary-Loss-Free Load Balancing
- FP8 Mixed Precision Support (geplant)
- Lightning Integration

#### GPT-2/GPT-3

**Dokumentation:** [docs/llm_modules.md](llm_modules.md)

Standard Transformer Language Model:

- Multi-Head Self-Attention
- Standard FFN Layers
- Configurable Layers & Heads
- GPT-2 Small/Medium/Large Configs

**Verwendung:**

```python
from aptt.modules.gpt import GPTModule

model = GPTModule(
    vocab_size=50000,
    d_model=768,
    n_layers=12,
    n_heads=12,
)
```

### Loss Functions & Heads

**Dokumentation:** [docs/llm_loss_head.md](llm_loss_head.md)

#### Language Modeling Loss

- Cross-Entropy mit Label Smoothing
- Perplexity Metric
- Token Accuracy Metric

#### Multi-Token Prediction Loss

- λ-weighted MTP Loss
- Multiple Future Token Predictions
- DeepSeek-V3 optimiert

#### Output Heads

- Standard LM Head mit Weight Tying
- Multi-Token Prediction Head
- Combined LM + MTP Head

### Mixture-of-Experts

**Dokumentation:** [docs/moe.md](moe.md)

Implementierung des DeepSeek-V3 MoE Systems:

#### Komponenten

- **FFNExpert**: SwiGLU-basierte Expert Networks
- **AuxiliaryLossFreeRouter**: Bias-basiertes Load Balancing
- **DeepSeekMoE**: Standard Implementation
- **EfficientDeepSeekMoE**: Optimierte Batch-Verarbeitung

**Features:**

- N_s=1 Shared Expert (immer aktiv)
- N_r=256 Routed Experts (selektiv)
- K_r=8 Experts pro Token
- Keine Auxiliary Loss erforderlich
- 2-3x schneller mit Efficient Implementation

### Text Datasets

**Dokumentation:** [docs/text_dataset.md](text_dataset.md)

Flexible Dataset Loader für Language Modeling:

#### Unterstützte Formate

- Raw Text Files (.txt)
- JSONL (.jsonl)
- Pre-tokenized (.pt)

#### Features

- Sliding Window mit konfigurierbarem Stride
- Multi-Token Prediction Targets
- Directory-based Loading
- Tokenizer-agnostic

**Verwendung:**

```python
from aptt.lightning_base.dataset import TextDataLoader

datamodule = TextDataLoader(
    train_data_path="train.txt",
    val_data_path="val.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
    batch_size=32,
)
```

## 👁️ Computer Vision

### Object Detection

#### YOLO (v3/v4/v5)

**Module:** `aptt.modul.yolo`

- YOLOv3, YOLOv4, YOLOv5 Architectures
- CSPDarknet Backbone
- FPN Neck
- Detection Head

**Verwendung:**

```python
from aptt.modul.yolo import YOLOModule

model = YOLOModule(
    num_classes=80,
    model_size="yolov5s",
    pretrained=True,
)
```

#### CenterNet

**Module:** `aptt.modul.centernet`

- Keypoint-based Detection
- ResNet/DLA Backbone
- Heatmap Prediction
- Multi-Scale Training

**Verwendung:**

```python
from aptt.modul.centernet import CenterNetModule

model = CenterNetModule(
    num_classes=80,
    backbone="resnet50",
)
```

### Feature Extractors

#### ResNet

- ResNet-18/34/50/101/152
- Pretrained Weights
- Feature Pyramid Support

#### DarkNet

- DarkNet-53
- CSPDarkNet
- YOLO Backbone

#### EfficientNet

- EfficientNet-B0 bis B7
- Compound Scaling
- MBConv Blocks

#### MobileNet

- MobileNetV2/V3
- Depthwise Separable Convolutions
- Efficient Mobile Inference

### Object Tracking

**Module:** `aptt.tracker`

- RNN-based Tracker
- ReID Encoder
- Multi-Object Tracking
- Track Management

**Features:**

- Kalman Filter Integration
- Appearance Features (ReID)
- Motion Prediction
- Track Association

## 🎵 Audio Processing

### Beamforming

**Module:** `aptt.model.beamforming`

- Multi-Channel Audio Processing
- Delay-and-Sum Beamforming
- MVDR (Minimum Variance Distortionless Response)
- Adaptive Beamforming

### Direction of Arrival (DOA)

**Module:** `aptt.model.doa`

- Acoustic Source Localization
- MUSIC Algorithm
- SRP-PHAT
- Neural DOA Estimation

### Complex-valued Networks

**Module:** `aptt.model.complex`

- Complex-valued Convolutions
- Complex Batch Normalization
- Phase-aware Processing
- Audio Enhancement

### WaveNet

**Module:** `aptt.model.feature.wavenet`

- Dilated Causal Convolutions
- Residual Connections
- Gated Activation Units
- Audio Generation

## ⚙️ Training & Utilities

### Continual Learning

**Module:** `aptt.lightning_base.continual_learning_manager`

- Knowledge Distillation
- Learning without Forgetting (LwF)
- Elastic Weight Consolidation (EWC)
- Progressive Neural Networks

### Callbacks

**Module:** `aptt.callbacks`

#### TensorRT Export

- Automatic TensorRT Optimization
- INT8 Quantization
- FP16 Mixed Precision

#### TorchScript Export

- Model Export for Production
- Graph Optimization
- Mobile Deployment

#### t-SNE Visualization

- Embedding Visualization
- Training Progress Monitoring
- Feature Space Analysis

#### Embedding Logger

- Log Embeddings to TensorBoard
- Dimensionality Reduction
- Cluster Analysis

### Metrics

**Module:** `aptt.metric`

#### Detection Metrics

- mAP (mean Average Precision)
- IoU (Intersection over Union)
- Precision/Recall

#### Language Model Metrics

- Perplexity
- Token Accuracy
- BLEU Score (geplant)

#### Audio Metrics

- SNR (Signal-to-Noise Ratio)
- RMSE (Root Mean Square Error)
- Mel Spectrogram Loss

## 📖 Beispiele

Alle Module haben vollständige Beispiele in `examples/`:

### Language Models

```bash
python examples/llm_modules_example.py      # GPT & DeepSeek-V3
python examples/llm_loss_head_example.py    # Loss Functions & Heads
python examples/moe_example.py              # Mixture-of-Experts
python examples/text_dataset_simple.py      # Text Datasets
```

### Computer Vision

```bash
python examples/yolo_example.py             # YOLO Detection
python examples/centernet_example.py        # CenterNet Detection
python examples/tracking_example.py         # Object Tracking
```

### Audio Processing

```bash
python examples/beamforming_example.py      # Beamforming
python examples/doa_example.py              # DOA Estimation
```

## 🔧 Entwicklung

### Neue Module Hinzufügen

1. **Erstelle Module:**

   ```python
   # src/aptt/modules/my_module.py
   from aptt.lightning_base.module import BaseModule

   class MyModule(BaseModule):
       def __init__(self, ...):
           super().__init__(...)
           # Your implementation
   ```

2. **Füge zu **init**.py hinzu:**

   ```python
   # src/aptt/modules/__init__.py
   from aptt.modules.my_module import MyModule

   __all__ = [..., "MyModule"]
   ```

3. **Erstelle Tests:**

   ```python
   # tests/test_my_module.py
   def test_my_module():
       model = MyModule(...)
       # Test implementation
   ```

4. **Dokumentation:**
   ```markdown
   # docs/my_module.md

   # My Module Documentation

   ...
   ```

### Code-Qualität

Alle Module müssen folgende Standards erfüllen:

- ✅ Ruff Code Style
- ✅ MyPy Type Hints (strict)
- ✅ Google-Style Docstrings
- ✅ Unit Tests
- ✅ Dokumentation

```bash
# Prüfung
ruff check src/aptt/modules/my_module.py
mypy src/aptt/modules/my_module.py
pytest tests/test_my_module.py
```

## 📚 Weiterführende Dokumentation

### Detaillierte Dokumentation

- [LLM Modules](llm_modules.md) - GPT & DeepSeek-V3
- [LLM Loss & Heads](llm_loss_head.md) - Language Modeling Losses
- [Mixture-of-Experts](moe.md) - MoE Implementation
- [Text Datasets](text_dataset.md) - Data Loading

### API Reference

- [Sphinx Documentation](../docs/_build/html/index.html) - Vollständige API
- [Examples](../examples/) - Code-Beispiele
- [Tests](../tests/) - Unit Tests

## 🆘 Hilfe & Support

Bei Fragen oder Problemen:

1. **Dokumentation:** Überprüfe die relevante Modul-Dokumentation
2. **Beispiele:** Siehe `examples/` für Verwendungsbeispiele
3. **Issues:** [GitHub Issues](https://github.com/afeldman/aptt/issues)
4. **Email:** anton.feldmann@gmail.com

## 🚀 Roadmap

### Geplante Module

#### Language Models

- [ ] LLaMA/LLaMA-2 Implementation
- [ ] BERT/RoBERTa Fine-Tuning
- [ ] LoRA/QLoRA für Efficient Fine-Tuning
- [ ] FP8 Mixed Precision für DeepSeek-V3

#### Computer Vision

- [ ] SAM (Segment Anything Model)
- [ ] DETR (Detection Transformer)
- [ ] ViT (Vision Transformer)
- [ ] Diffusion Models

#### Audio

- [ ] Whisper Speech Recognition
- [ ] MusicGen Audio Generation
- [ ] AudioLM

#### Multi-Modal

- [ ] CLIP Vision-Language
- [ ] Flamingo
- [ ] BLIP-2

---

**Stand:** Dezember 2024 | **Version:** 0.2.0
