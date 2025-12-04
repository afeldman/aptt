# Callbacks

PyTorch Lightning Callbacks für Export, Optimierung und Visualisierung.

## Module

### Export & Optimization

- **`torchscript.py`** - TorchScript-Export für Production Deployment
- **`tensor_rt.py`** - TensorRT-Optimierung für NVIDIA GPUs

### Logging & Visualization

- **`embedding_logger.py`** - Embedding-Visualisierung während des Trainings
- **`tsne_laplace_callback.py`** - t-SNE und Laplace Eigenmap Visualisierung

### Base

- **`base.py`** - Basis-Callback-Klasse und gemeinsame Funktionalität

## Verwendung

### TorchScript Export

```python
from aptt.callbacks import TorchScriptCallback

trainer = pl.Trainer(
    callbacks=[
        TorchScriptCallback(
            export_path="model.pt",
            method="script"  # oder "trace"
        )
    ]
)
```

### TensorRT Optimization

```python
from aptt.callbacks import TensorRTCallback

trainer = pl.Trainer(
    callbacks=[
        TensorRTCallback(
            export_path="model.engine",
            precision="fp16",
            workspace_size=1 << 30  # 1GB
        )
    ]
)
```

### Embedding Visualization

```python
from aptt.callbacks import EmbeddingLoggerCallback

trainer = pl.Trainer(
    callbacks=[
        EmbeddingLoggerCallback(
            log_every_n_epochs=5,
            num_samples=1000
        )
    ]
)
```

## Features

- ✅ Automatischer Export am Ende des Trainings
- ✅ Validierung der exportierten Modelle
- ✅ Integration mit TensorBoard und Weights & Biases
- ✅ Flexible Konfiguration per Callback-Parameter
- ✅ Fehlerbehandlung und Logging

## Weitere Informationen

Siehe Hauptdokumentation: [docs/modules_overview.md](../../../docs/modules_overview.md)
