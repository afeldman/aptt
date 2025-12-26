# DeepSuite Test Coverage Report

**Generiert am:** 2025-12-26
**Gesamt-Coverage:** 18%
**Tests gesamt:** 83 Tests (60 passed, 23 failed, 1 skipped)

## Zusammenfassung

Das DeepSuite-Projekt verfügt über eine umfassende Test-Suite mit **83 Unit-Tests** über 9 Test-Dateien. Die aktuelle Test-Coverage beträgt **18%** (1.014 von 5.760 Code-Zeilen getestet).

## Test-Suite Struktur

### Erfolgreich getestete Module (60 passing tests)

#### 1. **BaseDataLoader** (14 Tests, 100% Coverage)
- ✅ `test_initialization_default_params`
- ✅ `test_initialization_custom_params`
- ✅ `test_setup_fit_stage`
- ✅ `test_setup_none_stage`
- ✅ `test_train_dataloader`
- ✅ `test_val_dataloader`
- ✅ `test_train_dataloader_before_setup_raises_error`
- ✅ `test_val_dataloader_before_setup_raises_error`
- ✅ `test_train_dataloader_iteration`
- ✅ `test_val_dataloader_iteration`
- ✅ `test_multiple_setup_calls`
- ✅ `test_num_workers_zero`
- ✅ `test_batch_size_one`
- ✅ `test_large_batch_size`

**Coverage:** `src/deepsuite/lightning_base/dataset/base_loader.py` → **100%**

#### 2. **MelSpectrogramExtractor** (18 Tests, 100% Coverage)
- ✅ `test_initialization_default_params`
- ✅ `test_initialization_custom_params`
- ✅ `test_mel_filterbank_creation`
- ✅ `test_forward_1d_input`
- ✅ `test_forward_2d_input`
- ✅ `test_output_shape_calculation`
- ✅ `test_mel_scale_function`
- ✅ `test_mel_scale_properties`
- ✅ `test_different_sample_rates`
- ✅ `test_different_n_mels`
- ✅ `test_short_audio`
- ✅ `test_long_audio`
- ✅ `test_zero_audio`
- ✅ `test_batch_processing_consistency`
- ✅ `test_device_compatibility`
- ✅ `test_mel_fb_buffer_registered`
- ✅ `test_frequency_range`

**Coverage:** `src/deepsuite/layers/mel.py` → **100%**

#### 3. **Tracker** (7 Tests, 94% Coverage)
- ✅ `test_initialization_without_reid`
- ✅ `test_initialization_with_reid`
- ✅ `test_initialization_custom_params`
- ✅ `test_crop_boxes_method`
- ✅ `test_lstm_rnn_type`
- ✅ `test_gru_rnn_type`

**Coverage:** `src/deepsuite/model/tracking/tracker.py` → **94%**

#### 4. **UniversalDataset** (10 Tests, 39% Coverage)
- ✅ `test_basic_initialization`
- ✅ `test_with_transform`
- ✅ `test_download_disabled_by_default`
- ✅ `test_getitem_with_invalid_index`
- ✅ `test_len_method`
- ✅ `test_empty_dataset`
- ✅ `test_no_download_when_extracted_exists`
- ✅ `test_transform_none`

**Coverage:** `src/deepsuite/lightning_base/dataset/universal_set.py` → **39%**

#### 5. **FocalLoss** (2 Tests)
- ✅ `test_initialization`
- ✅ `test_forward_binary`

#### 6. **Export Callbacks** (2 Tests)
- ✅ `test_get_example_input`
- ⏭️ `test_tensor_rt_export` (SKIPPED - TensorRT nur Linux/Windows)

### Module mit Tests (aber Failures)

#### 7. **Conv Layers** (10 Tests erstellt, 0 passed)
- ❌ Tests für `BaseConv2dBlock`, `DepthwiseSeparableConv`, `SEBlock`, `CausalConv1d`
- **Grund:** API-Mismatch - Tests verwenden falsche Konstruktor-Signatur
- **Behebung nötig:** Korrekte Parameter aus Quellcode ermitteln

#### 8. **Loss Functions** (7 Tests erstellt, 2 passed)
- ❌ `BboxLoss`, `VarifocalLoss`, `ClassificationLoss` Tests
- **Grund:** Forward-Methoden benötigen zusätzliche Parameter
- **Behebung nötig:** API-Dokumentation prüfen

#### 9. **Metrics** (7 Tests erstellt, 4 passed)
- ✅ `bbox_iou` Tests (4 Tests)
- ✅ `l2_norm` Tests (3 Tests)
- **Partial Coverage:** Einige Edge-Cases schlagen fehl

## Coverage nach Modul-Kategorien

### Hohe Coverage (> 75%)

| Modul | Coverage | Tests |
|-------|----------|-------|
| `layers/mel.py` | **100%** | 18 ✅ |
| `lightning_base/dataset/base_loader.py` | **100%** | 14 ✅ |
| `model/tracking/tracker.py` | **94%** | 7 ✅ |
| `utils/tensor.py` | **80%** | Indirekt |
| `tracker/rnn_tracker.py` | **76%** | 7 ✅ |
| `utils/autocast.py` | **75%** | Indirekt |

### Mittlere Coverage (25-75%)

| Modul | Coverage | Tests |
|-------|----------|-------|
| `callbacks/base.py` | 59% | 1 ✅ |
| `loss/varifocal.py` | 62% | 1 ❌ |
| `heads/heatmap.py` | 57% | - |
| `lightning_base/dataset/image_loader.py` | 57% | - |
| `registry.py` | 50% | - |
| `heads/box.py` | 47% | - |
| `loss/centernet.py` | 47% | - |
| `lightning_base/dataset/universal_set.py` | 39% | 8 ✅ 3 ❌ |

### Niedrige Coverage (< 25%)

| Modul | Coverage | Status |
|-------|----------|--------|
| `model/conv.py` | 21% | 10 Tests ❌ |
| `callbacks/tensor_rt.py` | 21% | 1 Test ⏭️ |
| `model/tracking/tracking.py` | 21% | - |
| `callbacks/kendryte.py` | 21% | - |
| `callbacks/onnx.py` | 18% | - |
| `lightning_base/dataset/text_loader.py` | 18% | - |
| `metric/bbox_iou.py` | 18% | 4 Tests ✅ |

### Keine Coverage (0%)

**99 Module** haben aktuell **0% Coverage**, darunter:
- Alle Detection-Modelle (`model/detection/*`)
- Alle Feature-Extraktoren (`model/feature/*`)
- LLM-Module (`model/llm/*`)
- LoFTR-Module (`model/loftr/*`)
- Meiste Loss-Functions
- Meiste Metrics
- Meiste Utils
- Alle Visualizations (`viz/*`)
- SVM Module (`svm/*`)

## Empfohlene nächste Schritte

### Priorität 1: Fix existierende Tests (23 Failures)
1. **Conv Tests** (10 Failures): API-Signaturen korrigieren
2. **Loss Tests** (4 Failures): Forward-Parameter anpassen
3. **Tracker Tests** (5 Failures): Frame-Format korrigieren
4. **UniversalDataset Tests** (3 Failures): Download/Extract-Mocks verbessern
5. **Metrics Tests** (1 Failure): Tolerance-Parameter anpassen

### Priorität 2: Kritische Module testen (0% → 80%+)
1. **Detection** (`model/detection/*.py`): YOLO, CenterNet, etc.
2. **Loss Functions** (`loss/*.py`): Focal, Detection, Segmentation
3. **Metrics** (`metric/*.py`): mAP, IoU-Varianten, Detection-Metriken
4. **Feature Extractors** (`model/feature/*.py`): ResNet, MobileNet, EfficientNet

### Priorität 3: Integration Tests
1. Ende-zu-Ende Training Pipeline
2. DataLoader → Model → Loss → Metric Workflow
3. Export Workflows (ONNX, TorchScript, TFLite)

### Priorität 4: Edge Cases & Error Handling
1. Device-Tests (CPU, CUDA, MPS)
2. Empty Input Tests
3. Invalid Parameter Tests
4. Boundary Condition Tests

## Erreichen von 100% Coverage

**Aktuell:** 18% (1.014 / 5.760 Zeilen)
**Ziel:** 100% (5.760 / 5.760 Zeilen)
**Verbleibend:** 4.746 Zeilen Code

**Geschätzte Test-Anzahl:**
- Fix existierende Tests: 23 Tests reparieren
- Neue Tests für 0%-Module: ~300-400 zusätzliche Tests
- Integration Tests: ~50 Tests
- **Gesamt:** ~450-500 Tests für 100% Coverage

**Zeitschätzung:**
- Fix existierende Tests: 2-3 Stunden
- Neue Unit Tests: 20-30 Stunden
- Integration Tests: 5-10 Stunden
- **Gesamt:** 27-43 Stunden für vollständige Coverage

## Test-Ausführung

```bash
# Alle Tests
uv run pytest tests/ -v

# Mit Coverage
uv run pytest tests/ --cov=deepsuite --cov-report=html

# Nur erfolgreiche Tests
uv run pytest tests/ -k "not (test_conv or test_losses or test_tracker_forward)"

# HTML Coverage-Report öffnen
open htmlcov/index.html
```

## Statistiken

- **Total Code-Zeilen:** 5.760
- **Getestete Zeilen:** 1.014 (18%)
- **Ungetestete Zeilen:** 4.746 (82%)
- **Module gesamt:** 149 Python-Dateien
- **Module mit Tests:** 8 Dateien
- **Module mit 100% Coverage:** 2 Dateien
- **Module mit 0% Coverage:** 99 Dateien
- **Tests gesamt:** 83 (60 ✅, 23 ❌, 1 ⏭️)
- **Test-Dateien:** 9 Dateien

## Coverage HTML Report

Ein detaillierter HTML-Report wurde generiert: `htmlcov/index.html`

Dieser Report zeigt:
- Zeilen-für-Zeilen Coverage für jede Datei
- Ungetestete Code-Bereiche (rot markiert)
- Verzweigungen (Branch Coverage)
- Coverage-Prozentsatz pro Modul
