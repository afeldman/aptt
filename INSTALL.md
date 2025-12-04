# APTT Installation Guide

## Plattform-spezifische Installation

APTT unterstützt verschiedene Plattformen mit unterschiedlichen PyTorch-Backends.

### macOS Installation

```bash
# CPU-only (empfohlen für macOS)
uv sync --extra cpu

# Falls uv.lock Probleme macht, lösche ihn zuerst:
rm uv.lock
uv sync --extra cpu
```

**Hinweis:** TensorRT ist auf macOS nicht verfügbar. Der `TensorRTExportCallback` wird auf macOS einen Fehler werfen, wenn Sie versuchen ihn zu verwenden.

### Linux Installation

#### CPU-only

```bash
uv sync --extra cpu
```

#### CUDA 12.4 (empfohlen mit GPU)

```bash
uv sync --extra cu124
```

#### Mit TensorRT (nur Linux/Windows)

TensorRT ist als separates Package verfügbar und muss manuell installiert werden:

```bash
# Nach der Installation von aptt[cu124]:
uv pip install torch-tensorrt>=2.6.1

# Oder mit pip:
pip install torch-tensorrt>=2.6.1
```

**Hinweis:** TensorRT-Pakete sind nur für Linux und Windows verfügbar, nicht für macOS.

### Windows Installation

#### CPU-only

```bash
uv sync --extra cpu
```

#### CUDA 12.4

```bash
uv sync --extra cu124
```

#### Mit TensorRT (nur Linux/Windows)

```bash
# Nach der Installation von aptt[cu124]:
uv pip install torch-tensorrt>=2.6.1

# Oder mit pip:
pip install torch-tensorrt>=2.6.1
```

**Hinweis:** TensorRT-Pakete sind nur für Linux und Windows verfügbar, nicht für macOS.

## Development Installation

Für Entwicklung mit zusätzlichen Tools:

```bash
# macOS
uv sync --extra cpu --extra dev

# Linux/Windows mit CUDA
uv sync --extra cu124 --extra dev
```

## Dokumentation erstellen

```bash
uv sync --extra doc
cd docs
make html
```

## Bekannte Probleme

### uv.lock enthält TensorRT auf macOS

Wenn Sie auf macOS Fehler sehen wie:

```
RuntimeError: TensorRT currently only builds wheels for Linux and Windows
```

**Lösung 1:** Lockfile neu erstellen (empfohlen)

```bash
rm uv.lock
uv sync --extra cpu
```

**Lösung 2:** Alternative Installation

```bash
# Installiere erst PyTorch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dann aptt ohne torch-tensorrt
uv pip install -e .
```

### TensorRT-Callback auf macOS

Der `TensorRTExportCallback` prüft automatisch die Plattform und wirft eine hilfreiche Fehlermeldung:

```python
from aptt.callbacks import TensorRTExportCallback

# Auf macOS:
callback = TensorRTExportCallback()
# ❌ RuntimeError: TensorRT is not available on darwin.
#    Only supported on Linux and Windows.
```

**Lösung:** Verwenden Sie stattdessen `TorchScriptExportCallback`:

```python
from aptt.callbacks import TorchScriptExportCallback

callback = TorchScriptExportCallback()  # ✅ Funktioniert auf allen Plattformen
```

## Verifizierung

Testen Sie die Installation:

```python
import torch
import aptt

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"APTT: {aptt.__version__}")

# Teste Import der Hauptmodule
from aptt import Yolo, CenterNetModule, BaseModule
print("✅ Installation erfolgreich!")
```

## Package Manager Wahl

### uv (empfohlen)

```bash
uv sync --extra cpu  # oder cu124
```

### pip

```bash
pip install -e ".[cpu]"  # oder [cu124]
```

### conda/mamba

```bash
# PyTorch zuerst
conda install pytorch torchvision torchaudio -c pytorch

# Dann aptt
pip install -e .
```

## Minimal-Installation (nur Core)

Wenn Sie nur die Basis-Dependencies ohne PyTorch brauchen:

```bash
uv pip install -e . --no-deps
# Dann manuell fehlende Packages hinzufügen
```

## Support

- **Plattformen:** macOS (CPU), Linux (CPU/CUDA), Windows (CPU/CUDA)
- **Python:** >=3.11
- **PyTorch:** >=2.6.0
- **TensorRT:** Nur Linux/Windows mit CUDA

Bei Problemen: [GitHub Issues](https://github.com/afeldman/aptt/issues)
