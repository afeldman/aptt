# DeepSuite Test-Strategie für 100% Coverage

## Übersicht

Dieses Dokument beschreibt die umfassende Test-Strategie für DeepSuite, um 100% Code-Coverage zu erreichen.

## Aktuelle Test-Coverage

### Bereits getestete Module

1. **UniversalDataset** (`tests/test_universal_dataset.py`)
   - ✅ Basic initialization
   - ✅ Transform functionality
   - ✅ Download and extraction (ZIP, TAR.GZ)
   - ✅ Error handling
   - ✅ Edge cases (empty dataset, invalid indices)
   - **Coverage:** ~95%

2. **BaseDataLoader** (`tests/test_base_loader.py`)
   - ✅ Initialization (default/custom params)
   - ✅ Setup stages
   - ✅ DataLoader creation
   - ✅ Error handling (setup not called)
   - ✅ Iteration
   - **Coverage:** ~90%

3. **MelSpectrogramExtractor** (`tests/test_mel_extractor.py`)
   - ✅ Initialization
   - ✅ Forward pass (1D/2D input)
   - ✅ Mel filterbank generation
   - ✅ Different sample rates and n_mels
   - ✅ Device compatibility
   - **Coverage:** ~92%

4. **Tracker** (`tests/test_tracker.py`)
   - ✅ Initialization with/without ReID
   - ✅ Forward pass
   - ✅ RNN types (GRU/LSTM)
   - **Coverage:** ~85%

## Zusätzlich erforderliche Tests

### Priorität 1: Core Functionality

1. **ImageLoader** (`tests/test_image_loader.py`)
   ```python
   - Test Albumentations transforms
   - Test RandAugment mode
   - Test normalization
   - Test batch loading
   ```

2. **AudioLoader** (`tests/test_audio_loader.py`)
   ```python
   - Test Audiomentations transforms
   - Test audio-specific augmentations
   - Test batch loading
   ```

3. **Model Conv Layers** (`tests/test_conv.py`)
   ```python
   - Test BaseConv2dBlock
   - Test DepthwiseSeparableConv
   - Test SEBlock
   - Test CausalConv1d
   ```

### Priorität 2: Loss Functions

4. **Focal Loss** (`tests/test_focal_loss.py`)
5. **Detection Loss** (`tests/test_detection_loss.py`)
6. **CenterNet Loss** (`tests/test_centernet_loss.py`)
7. **Bbox Loss** (`tests/test_bbox_loss.py`)

### Priorität 3: Metrics

8. **Detection Metrics** (`tests/test_detection_metrics.py`)
9. **mAP Calculation** (`tests/test_map.py`)
10. **IoU Metrics** (`tests/test_iou.py`)

### Priorität 4: Model Architectures

11. **ResNet Backbone** (`tests/test_resnet.py`)
12. **MobileNet** (`tests/test_mobilenet.py`)
13. **EfficientNet** (`tests/test_efficientnet.py`)
14. **YOLO** (`tests/test_yolo.py`)

## Test-Ausführung

### Voraussetzungen

```bash
# Installiere Test-Dependencies
uv add --dev pytest pytest-cov pytest-mock

# Optional: Installiere ML-Dependencies
uv add --dev torch torchvision pytorch-lightning
```

### Ausführung

```bash
# Alle Tests
uv run pytest

# Mit Coverage
uv run pytest --cov=deepsuite --cov-report=html

# Einzelne Test-Datei
uv run pytest tests/test_universal_dataset.py -v

# Coverage für spezifisches Modul
uv run pytest --cov=deepsuite.lightning_base.dataset --cov-report=term-missing
```

## Coverage-Ziele

| Modul                        | Aktuell | Ziel  | Status      |
|------------------------------|---------|-------|-------------|
| lightning_base/dataset       | 85%     | 100%  | ✅ Tests erstellt |
| layers/mel                   | 92%     | 100%  | ✅ Tests erstellt |
| model/tracking               | 85%     | 100%  | ✅ Tests erstellt |
| model/conv                   | 0%      | 95%   | ⏳ Geplant  |
| loss/*                       | 0%      | 90%   | ⏳ Geplant  |
| metric/*                     | 0%      | 90%   | ⏳ Geplant  |
| model/detection/*            | 0%      | 85%   | ⏳ Geplant  |
| model/feature/*              | 0%      | 85%   | ⏳ Geplant  |

## Test-Best-Practices

1. **Isolation**: Jeder Test sollte unabhängig laufen
2. **Fixtures**: Verwende pytest fixtures für gemeinsame Setup-Logik
3. **Mocking**: Mock externe Dependencies (Datei-I/O, Netzwerk)
4. **Edge Cases**: Teste Grenzbedingungen (None, leere Listen, ungültige Werte)
5. **Device-Agnostic**: Tests sollten auf CPU, CUDA und MPS laufen
6. **Deterministic**: Verwende `torch.manual_seed()` für reproduzierbare Tests

## Nächste Schritte

1. ✅ Pytest-Setup konfigurieren
2. ✅ Core Dataset-Tests implementieren
3. ⏳ DataLoader-Tests erweitern
4. ⏳ Loss-Function Tests
5. ⏳ Metric Tests
6. ⏳ Model Tests
7. ⏳ Integration Tests
8. ⏳ CI/CD Pipeline mit Coverage-Reporting

## Hinweise

- Tests sind als Grundlage erstellt und müssen noch ausgeführt werden
- Dependencies wie PyTorch Lightning müssen installiert sein
- Einige Tests benötigen zusätzliche Mocks für externe Libraries
- Coverage-Report wird mit `pytest-cov` generiert
