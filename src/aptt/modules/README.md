# Modules

PyTorch Lightning Module für verschiedene Deep Learning Tasks.

## Module

### Language Models

- **`gpt.py`** - GPT-2/GPT-3 Transformer-Architektur
- **`deepseek.py`** - DeepSeek-V3 mit MLA und MoE

### Object Detection

- **`yolo.py`** - YOLO (v3/v4/v5) Object Detection
- **`centernet.py`** - CenterNet Anchor-free Detection

### Object Tracking

- **`tracking.py`** - RNN-basiertes Multi-Object Tracking mit ReID

## Verwendung

### GPT Module

```python
from aptt.modules import GPTModule
import pytorch_lightning as pl

model = GPTModule(
    vocab_size=50000,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072,
    dropout=0.1,
    learning_rate=1e-4
)

trainer = pl.Trainer(max_steps=100000, accelerator="gpu")
trainer.fit(model, datamodule)
```

### DeepSeek-V3 Module

```python
from aptt.modules import DeepSeekModule

model = DeepSeekModule(
    vocab_size=50000,
    d_model=2048,
    n_layers=24,
    n_heads=16,
    # MLA parameters
    d_h_c=256,              # Compressed KV dimension
    d_h_r=64,               # RoPE dimension
    # MoE parameters
    use_moe=True,
    n_shared_experts=1,
    n_routed_experts=256,
    n_expert_per_token=8,
    # MTP parameters
    use_mtp=True,
    mtp_depth=3,
    mtp_lambda=0.3,
    # Training
    learning_rate=1e-4,
    weight_decay=0.1
)

trainer = pl.Trainer(
    max_steps=100000,
    accelerator="gpu",
    devices=8,
    strategy="deepspeed_stage_2"
)
trainer.fit(model, datamodule)
```

### YOLO Module

```python
from aptt.modules import YOLOModule

model = YOLOModule(
    num_classes=80,
    model_size="yolov5s",  # 's', 'm', 'l', 'x'
    pretrained=True,
    img_size=640,
    conf_thres=0.25,
    iou_thres=0.45,
    learning_rate=1e-3
)

trainer = pl.Trainer(max_epochs=100, accelerator="gpu")
trainer.fit(model, datamodule)
```

### CenterNet Module

```python
from aptt.modules import CenterNetModule

model = CenterNetModule(
    num_classes=80,
    backbone="resnet50",
    head_conv=64,
    learning_rate=1e-4
)

trainer = pl.Trainer(max_epochs=140, accelerator="gpu")
trainer.fit(model, datamodule)
```

### Tracking Module

```python
from aptt.modules import TrackingModule

model = TrackingModule(
    feature_dim=512,
    hidden_dim=256,
    num_layers=2,
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)

# Inference
detections = detector(frame)
tracks = model.update(detections)
```

## Module-Funktionalität

### Alle Lightning Modules bieten

- ✅ **Training Step**: Automatische Loss-Berechnung
- ✅ **Validation Step**: Metriken-Logging
- ✅ **Test Step**: Evaluation auf Test-Set
- ✅ **Optimizer Config**: Adam/AdamW mit Scheduler
- ✅ **Logging**: Integration mit TensorBoard/WandB
- ✅ **Checkpointing**: Automatisches Model Saving

### Beispiel: Training Loop

```python
import pytorch_lightning as pl
from aptt.modules import DeepSeekModule
from aptt.lightning_base.dataset import TextDataLoader

# Data
datamodule = TextDataLoader(
    train_data_path="train.txt",
    val_data_path="val.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
    batch_size=32
)

# Model
model = DeepSeekModule(
    vocab_size=50000,
    d_model=2048,
    use_moe=True,
    use_mtp=True
)

# Callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',
    mode='min',
    save_top_k=3,
    filename='deepseek-{epoch:02d}-{val_loss:.2f}'
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# Training
trainer = pl.Trainer(
    max_steps=100000,
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    callbacks=[checkpoint_callback, lr_monitor],
    log_every_n_steps=10,
    val_check_interval=1000
)

trainer.fit(model, datamodule)
```

## Module-Hierarchie

```
modules/
├── gpt.py           # GPT-2/GPT-3
├── deepseek.py      # DeepSeek-V3 (MLA + MoE + MTP)
├── yolo.py          # YOLO Object Detection
├── centernet.py     # CenterNet Detection
└── tracking.py      # Multi-Object Tracking
```

## Model Zoo

| Module | Parameters | Training Time | Inference Speed |
|--------|-----------|---------------|-----------------|
| GPT-Small | 124M | ~3 days (8x A100) | 50 tokens/s |
| DeepSeek-Small | 51M | ~1 day (8x A100) | 30 tokens/s |
| DeepSeek-Base | 1.3B | ~2 weeks (64x A100) | 15 tokens/s |
| YOLOv5s | 7.2M | ~12 hours (1x V100) | 140 FPS |
| YOLOv5m | 21M | ~1 day (1x V100) | 100 FPS |
| CenterNet-R50 | 32M | ~2 days (4x V100) | 45 FPS |

## Weitere Informationen

- Hauptdokumentation: [docs/modules_overview.md](../../../docs/modules_overview.md)
- LLM Modules: [docs/llm_modules.md](../../../docs/llm_modules.md)
- MoE: [docs/moe.md](../../../docs/moe.md)
- Beispiele: [examples/](../../../examples/)
