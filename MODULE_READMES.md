# Module READMEs Übersicht

Alle Module in `src/aptt` haben jetzt eigene README-Dateien zur besseren Navigation und Dokumentation.

## ✅ Erstellte READMEs

### Hauptverzeichnis
- **[src/aptt/README.md](src/aptt/README.md)** - Übersicht über alle Module mit Schnellzugriff

### Core Module

1. **[callbacks/README.md](src/aptt/callbacks/README.md)**
   - Export & Optimization (TorchScript, TensorRT)
   - Logging & Visualization (Embedding Logger, t-SNE)
   - Beispiele für alle Callbacks

2. **[heads/README.md](src/aptt/heads/README.md)**
   - Language Model Heads (LM, MTP)
   - Computer Vision Heads (Classification, Detection, CenterNet)
   - Head-Loss Paarungen

3. **[layers/README.md](src/aptt/layers/README.md)**
   - Attention Mechanisms (MLA, RoPE, KV-Compression)
   - Mixture-of-Experts (MoE)
   - Specialized Layers (Complex, Hermite, Laguerre)

4. **[lightning_base/README.md](src/aptt/lightning_base/README.md)**
   - Base Lightning Module
   - Dataset Loaders (Text, Image, Audio)
   - Continual Learning Manager

5. **[loss/README.md](src/aptt/loss/README.md)**
   - Language Model Losses (Cross-Entropy, MTP)
   - Object Detection Losses (Focal, GIoU, DFL)
   - Knowledge Distillation (Distillation, LwF)

6. **[metric/README.md](src/aptt/metric/README.md)**
   - Object Detection Metrics (mAP, Precision, Recall)
   - Confusion Matrix
   - IoU Variants (GIoU, DIoU, CIoU)

7. **[model/README.md](src/aptt/model/README.md)**
   - Language Models (GPT, DeepSeek-V3)
   - Object Detection (YOLO, CenterNet, EfficientDet)
   - Feature Extraction (ResNet, EfficientNet, DarkNet, FPN)
   - Audio Processing (Beamforming, DOA)

8. **[modules/README.md](src/aptt/modules/README.md)**
   - PyTorch Lightning Modules
   - GPT, DeepSeek-V3, YOLO, CenterNet, Tracking
   - Training Examples mit Lightning

9. **[tracker/README.md](src/aptt/tracker/README.md)**
   - Multi-Object Tracking (SORT, DeepSORT, ByteTrack)
   - Re-Identification
   - Tracking Pipeline & Evaluation

10. **[utils/README.md](src/appt/utils/README.md)**
    - Bounding Box Operations
    - Image & Tensor Processing
    - Device Management
    - Signal Processing

11. **[viz/README.md](src/aptt/viz/README.md)**
    - Embedding Visualization (t-SNE, UMAP, PCA)
    - Training Curves
    - Attention Weights

## 📚 Dokumentations-Hierarchie

```
APTT/
├── README.md                          # Hauptprojekt-README
├── MODULE_READMES.md                  # Diese Datei
│
├── docs/                              # Detaillierte Dokumentation
│   ├── modules_overview.md            # Komplette Modulübersicht
│   ├── llm_modules.md                 # LLM Dokumentation
│   ├── llm_loss_head.md               # LLM Losses & Heads
│   ├── moe.md                         # Mixture-of-Experts
│   └── text_dataset.md                # Text Datasets
│
├── src/aptt/                          # Source Code
│   ├── README.md                      # Modul-Übersicht mit Imports
│   │
│   ├── callbacks/README.md            # Training Callbacks
│   ├── heads/README.md                # Output Heads
│   ├── layers/README.md               # Neural Network Layers
│   ├── lightning_base/README.md       # Lightning Base
│   ├── loss/README.md                 # Loss Functions
│   ├── metric/README.md               # Evaluation Metrics
│   ├── model/README.md                # Model Architectures
│   ├── modules/README.md              # Lightning Modules
│   ├── tracker/README.md              # Object Tracking
│   ├── utils/README.md                # Utility Functions
│   └── viz/README.md                  # Visualization
│
└── examples/                          # Code-Beispiele
    ├── llm_modules_example.py
    ├── llm_loss_head_example.py
    ├── moe_example.py
    └── text_dataset_simple.py
```

## 🎯 Verwendung

### Für Entwickler

1. **Neues Modul verstehen**: Lies das entsprechende README im Modulverzeichnis
2. **API-Referenz**: Siehe die Beispiele in jedem README
3. **Integration**: Import-Beispiele und Code-Snippets in jedem README

### Für Nutzer

1. **Quick Start**: [README.md](README.md) - Hauptprojekt-Übersicht
2. **Module finden**: [src/aptt/README.md](src/aptt/README.md) - Schnellzugriff
3. **Details**: Module-spezifische READMEs für tiefergehende Info

### Navigation

```bash
# Von Root zu Modul
cd src/aptt/layers
cat README.md

# Liste alle Module
find src/aptt -name "README.md" -type f

# Suche nach Keyword
grep -r "Multi-Head Latent Attention" src/aptt/*/README.md
```

## �� Statistiken

- **Anzahl Module-READMEs**: 12
- **Gesamte Zeilen Dokumentation**: ~4500+ Zeilen
- **Abgedeckte Module**: 100%
- **Code-Beispiele**: ~100+

## ✨ Features aller READMEs

Jedes README enthält:

- ✅ **Modulübersicht** - Was ist im Modul enthalten
- ✅ **Verwendungsbeispiele** - Praktische Code-Snippets
- ✅ **API-Referenz** - Wichtigste Klassen und Funktionen
- ✅ **Features** - Hauptfunktionalität
- ✅ **Best Practices** - Empfohlene Verwendung
- ✅ **Links** - Verweise auf weitere Dokumentation

## 🔗 Weitere Informationen

- **Projekt-Homepage**: [README.md](README.md)
- **Vollständige Dokumentation**: [docs/](docs/)
- **Beispiele**: [examples/](examples/)
- **Tests**: [tests/](tests/)

---

**Version**: 0.2.0 | **Stand**: 4. Dezember 2025
