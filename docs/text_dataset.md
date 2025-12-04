# Text/LLM Dataset Loader

Dataset und DataLoader für Language Modeling mit GPT und DeepSeek-V3.

## Überblick

Das `TextDataset` und `TextDataLoader` ermöglichen das Laden und Verarbeiten von Text-Daten für Language Model Training:

- **Flexible Formate**: .txt, .jsonl, .pt (pre-tokenized)
- **Sliding Window**: Effiziente Nutzung langer Texte
- **Multi-Token Prediction**: Optional für DeepSeek-V3
- **PyTorch Lightning**: Native Integration

## Komponenten

### 1. TextDataset

Dataset-Klasse für tokenisierte Text-Daten.

```python
from aptt.lightning_base.dataset import TextDataset

dataset = TextDataset(
    data_path="data/train.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
    stride=256,
    return_mtp=False,
    mtp_depth=1,
)

# Access samples
sample = dataset[0]
print(sample["input_ids"].shape)  # (512,)
print(sample["labels"].shape)     # (512,)
```

**Parameter:**

- `data_path`: Path zu Daten (file oder directory)
- `tokenizer`: Tokenizer mit `encode()` oder `__call__()` method
- `max_seq_len`: Maximale Sequenzlänge (default: 512)
- `stride`: Sliding window stride (default: max_seq_len)
- `return_mtp`: Multi-Token Prediction aktivieren (default: False)
- `mtp_depth`: MTP lookahead depth (default: 1)

**Unterstützte Formate:**

1. **Raw Text (.txt)**:

   ```
   This is training data.
   Multiple lines are supported.
   ```

2. **JSONL (.jsonl)**:

   ```json
   {"text": "First document"}
   {"text": "Second document"}
   ```

3. **Pre-tokenized (.pt)**:
   ```python
   token_ids = torch.tensor([1, 2, 3, ...])
   torch.save(token_ids, "data.pt")
   ```

### 2. TextDataLoader

LightningDataModule für Train/Val/Test Splits.

```python
from aptt.lightning_base.dataset import TextDataLoader

datamodule = TextDataLoader(
    train_data_path="data/train.txt",
    val_data_path="data/val.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
    batch_size=32,
    num_workers=4,
)

# Setup und verwenden
datamodule.setup("fit")
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
```

**Parameter:**

- `train_data_path`: Path zu Training-Daten
- `val_data_path`: Path zu Validation-Daten (optional)
- `test_data_path`: Path zu Test-Daten (optional)
- `tokenizer`: Tokenizer-Instanz
- `max_seq_len`: Maximale Sequenzlänge
- `batch_size`: Batch size
- `num_workers`: DataLoader workers
- `stride`: Sliding window stride
- `return_mtp`: MTP aktivieren
- `mtp_depth`: MTP depth

## Tokenizer

Das Dataset unterstützt verschiedene Tokenizer:

### HuggingFace Tokenizers

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

dataset = TextDataset(
    data_path="data.txt",
    tokenizer=tokenizer,  # Has .encode().ids
    max_seq_len=512,
)
```

### Transformers Tokenizers

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = TextDataset(
    data_path="data.txt",
    tokenizer=tokenizer,  # Has .encode()
    max_seq_len=512,
)
```

### Custom Tokenizer

```python
class MyTokenizer:
    def encode(self, text: str) -> list[int]:
        # Custom tokenization
        return [...]

tokenizer = MyTokenizer()

dataset = TextDataset(
    data_path="data.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
)
```

## Sliding Window

Lange Texte werden mit einem Sliding Window in Chunks aufgeteilt:

```
Text: [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]
max_seq_len = 4, stride = 2

Samples:
  [T0, T1, T2, T3] → [T1, T2, T3, T4]  (labels)
  [T2, T3, T4, T5] → [T3, T4, T5, T6]
  [T4, T5, T6, T7] → [T5, T6, T7, T8]
  ...
```

**Overlap vs. No Overlap:**

- Training: `stride < max_seq_len` (z.B. stride = max_seq_len // 2)
- Validation: `stride = max_seq_len` (kein Overlap)

## Multi-Token Prediction (MTP)

Für DeepSeek-V3 können zusätzliche Future-Token-Targets generiert werden:

```python
dataset = TextDataset(
    data_path="data.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
    return_mtp=True,
    mtp_depth=3,  # Predict 1, 2, 3 tokens ahead
)

sample = dataset[0]
print(sample["input_ids"].shape)    # (512,)
print(sample["labels"].shape)       # (512,)
print(sample["mtp_labels"].shape)   # (3, 512)
```

**MTP Targets:**

```
Input:      [T0, T1, T2, T3, T4, ...]
Labels:     [T1, T2, T3, T4, T5, ...]
MTP[0]:     [T2, T3, T4, T5, T6, ...]  (1 ahead)
MTP[1]:     [T3, T4, T5, T6, T7, ...]  (2 ahead)
MTP[2]:     [T4, T5, T6, T7, T8, ...]  (3 ahead)
```

## Training mit DeepSeek-V3

```python
from aptt.lightning_base.dataset import TextDataLoader
from aptt.modules.deepseek import DeepSeekModule
import pytorch_lightning as pl

# Dataset mit MTP
datamodule = TextDataLoader(
    train_data_path="data/train.txt",
    val_data_path="data/val.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
    batch_size=32,
    return_mtp=True,  # Für DeepSeek
    mtp_depth=1,
)

# Modell
model = DeepSeekModule(
    vocab_size=50000,
    d_model=2048,
    n_layers=24,
    n_heads=16,
    use_mtp=True,  # Aktiviere MTP Loss
    mtp_lambda=0.3,
)

# Training
trainer = pl.Trainer(
    max_steps=100000,
    accelerator="gpu",
    devices=1,
)

trainer.fit(model, datamodule)
```

## Training mit GPT

```python
from aptt.lightning_base.dataset import TextDataLoader
from aptt.modules.gpt import GPTModule
import pytorch_lightning as pl

# Dataset ohne MTP
datamodule = TextDataLoader(
    train_data_path="data/train.txt",
    val_data_path="data/val.txt",
    tokenizer=tokenizer,
    max_seq_len=1024,
    batch_size=16,
    return_mtp=False,  # Standard LM
)

# Modell
model = GPTModule(
    vocab_size=50000,
    d_model=768,
    n_layers=12,
    n_heads=12,
)

# Training
trainer = pl.Trainer(
    max_steps=100000,
    accelerator="gpu",
    devices=1,
)

trainer.fit(model, datamodule)
```

## Batch Structure

**Standard (GPT):**

```python
batch = {
    "input_ids": Tensor[batch_size, max_seq_len],
    "labels": Tensor[batch_size, max_seq_len],
}
```

**Mit MTP (DeepSeek):**

```python
batch = {
    "input_ids": Tensor[batch_size, max_seq_len],
    "labels": Tensor[batch_size, max_seq_len],
    "mtp_labels": Tensor[batch_size, mtp_depth, max_seq_len],
}
```

## Best Practices

### 1. Sequenzlänge

- **GPT-2**: 1024 tokens
- **GPT-3**: 2048 tokens
- **DeepSeek-V3**: 4096-8192 tokens

Kleinere max_seq_len für schnelleres Training:

```python
# Development
max_seq_len=512

# Production
max_seq_len=2048
```

### 2. Stride

- Training: `stride = max_seq_len // 2` (50% overlap)
- Validation: `stride = max_seq_len` (kein overlap)

### 3. Batch Size

Abhängig von GPU Memory:

- 24GB GPU: batch_size=32, max_seq_len=512
- 48GB GPU: batch_size=64, max_seq_len=1024

### 4. Tokenizer

Production-ready Tokenizer:

- **BPE**: HuggingFace `tokenizers`
- **SentencePiece**: Google SentencePiece
- **Tiktoken**: OpenAI Tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
# Train on your corpus
```

### 5. Data Format

Empfohlen:

1. **Pre-tokenized .pt**: Schnellstes Loading
2. **JSONL**: Flexible Metadaten
3. **Raw .txt**: Einfachste Variante

## Beispiele

Siehe vollständige Beispiele in:

- `examples/text_dataset_simple.py`: Einfaches Beispiel mit Char-Tokenizer
- `examples/text_dataset_example.py`: Erweitertes Beispiel mit BPE

### Quick Start

```bash
# Simple Example (keine Dependencies)
python examples/text_dataset_simple.py

# Mit tokenizers library
pip install tokenizers
python examples/text_dataset_example.py
```

## Performance

### Loading Speed

- **Pre-tokenized (.pt)**: ~100k samples/sec
- **JSONL**: ~50k samples/sec
- **Raw Text**: ~20k samples/sec (abhängig von Tokenizer)

### Memory

Dataset hält alle Samples im RAM:

- 1M samples × 512 tokens × 8 bytes ≈ 4GB RAM

Für große Datasets: Streaming oder Memory-Mapped Files.

## Erweiterungen

### Custom Transforms

```python
class MyTextDataset(TextDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # Custom preprocessing
        sample["attention_mask"] = (sample["input_ids"] != 0).long()
        return sample
```

### Multi-File Loading

```python
# Load all files in directory
dataset = TextDataset(
    data_path="data/corpus/",  # Directory
    tokenizer=tokenizer,
    max_seq_len=512,
)
```

### Dynamic Batching

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Dynamic padding
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    # ...
    return {"input_ids": input_ids, ...}

dataloader = DataLoader(dataset, collate_fn=collate_fn)
```

## Referenzen

- **Related**: `docs/llm_modules.md` für Model-Dokumentation
- **Examples**: `examples/text_dataset_simple.py`
- **Code**: `src/aptt/lightning_base/dataset/text_loader.py`
