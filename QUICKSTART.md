# DeepSuite - Quick Reference Guide

## 📚 Dataset Modules Documentation

### Quick Start

#### 1. UniversalDataset (Auto-Download)
```python
from deepsuite.lightning_base.dataset.universal_set import UniversalDataset

# Download and extract dataset automatically
dataset = UniversalDataset(
    download_url="https://example.com/data.zip",
    root_dir="./data",
    auto_download=True
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)
```

#### 2. ImageLoader (Computer Vision)
```python
from deepsuite.lightning_base.dataset.image_loader import ImageLoader

# Create data module
datamodule = ImageLoader(
    batch_size=32,
    num_workers=4,
    randaugment=True,  # aggressive augmentation
    image_size=(224, 224)
)

# Setup and use with trainer
datamodule.setup()
trainer = Trainer()
trainer.fit(model, datamodule)
```

#### 3. AudioLoader (Audio Processing)
```python
from deepsuite.lightning_base.dataset.audio_loader import AudioLoader

# Create audio data module
datamodule = AudioLoader(
    batch_size=64,
    num_workers=8,
    randaugment=True  # noise, pitch shift, time stretch, etc.
)

datamodule.setup()
```

#### 4. TextDataset (Language Models)
```python
from deepsuite.lightning_base.dataset.text_loader import TextDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = TextDataset(
    data_path="data/corpus.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
    stride=256,
    return_mtp=True,  # Multi-Token Prediction
    mtp_depth=4
)

sample = dataset[0]  # {"input_ids": Tensor, "labels": Tensor}
```

---

## 🛠️ Code Quality Commands

### Run All Checks
```bash
# Check ruff + mypy
uv run ruff check src/deepsuite/lightning_base/dataset/ && \
uv run mypy src/deepsuite/lightning_base/dataset/ --ignore-missing-imports
```

### Ruff Only
```bash
# Check
uv run ruff check src/deepsuite/lightning_base/dataset/

# Fix automatically
uv run ruff check src/deepsuite/lightning_base/dataset/ --fix

# Fix including unsafe fixes
uv run ruff check src/deepsuite/lightning_base/dataset/ --fix --unsafe-fixes
```

### MyPy Only
```bash
# Check types
uv run mypy src/deepsuite/lightning_base/dataset/ --ignore-missing-imports

# With error codes
uv run mypy src/deepsuite/lightning_base/dataset/ --show-error-codes --ignore-missing-imports

# Specific file
uv run mypy src/deepsuite/lightning_base/dataset/universal_set.py
```

---

## 📖 API Reference

### UniversalDataset

```python
class UniversalDataset(Dataset[tuple[Any, Any]]):
    def __init__(
        self,
        data: list[Any] | None = None,
        labels: list[Any] | None = None,
        transform: Any | None = None,
        download_url: str | None = None,
        root_dir: str | None = None,
        auto_download: bool = False
    ) -> None:
        """Initialize dataset with optional auto-download."""

    def __len__(self) -> int:
        """Return number of samples."""

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Get sample and label at index."""
```

**Supported Formats:**
- `.zip` - ZIP archives
- `.tar.gz`, `.tgz` - Gzipped TAR
- `.tar` - TAR archives

---

### BaseDataLoader

```python
class BaseDataLoader(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        randaugment: bool = False
    ) -> None:
        """Initialize data loader."""

    @abstractmethod
    def _get_train_transforms(self) -> Any:
        """Return training augmentation pipeline."""

    @abstractmethod
    def _get_val_transforms(self) -> Any:
        """Return validation preprocessing."""

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Initialize train/val datasets."""

    def train_dataloader(self) -> DataLoader[Any]:
        """Return training DataLoader."""

    def val_dataloader(self) -> DataLoader[Any]:
        """Return validation DataLoader."""
```

---

### ImageLoader

```python
class ImageLoader(BaseDataLoader):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        randaugment: bool = False,
        image_size: tuple[int, int] = (256, 256)
    ) -> None:
        """Initialize image data loader."""

    def _get_train_transforms(self) -> A.Compose:
        """Return training augmentations (Albumentations)."""

    def _get_val_transforms(self) -> A.Compose:
        """Return validation preprocessing."""
```

**Augmentations (Standard):**
- Horizontal flip (50%)
- Random brightness/contrast (20%)
- ImageNet normalization

**Augmentations (RandAugment):**
- Shifts, scales, rotations
- Coarse dropout
- Color jittering
- Gaussian noise

---

### AudioLoader

```python
class AudioLoader(BaseDataLoader):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        randaugment: bool = False
    ) -> None:
        """Initialize audio data loader."""

    def _get_train_transforms(self) -> Any:
        """Return training augmentations (Audiomentations)."""

    def _get_val_transforms(self) -> Any:
        """Return validation preprocessing."""
```

**Augmentations (Standard):**
- Gaussian noise
- Normalization

**Augmentations (RandAugment):**
- Time stretching
- Pitch shifting
- Clipping distortion
- Gain adjustment
- Random shifts

---

## 🎯 Best Practices

### Type Hints
```python
# ✅ Correct
from torch.utils.data import Dataset, DataLoader
from typing import Any

class MyDataset(Dataset[tuple[Any, Any]]):
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        ...

def my_loader(self) -> DataLoader[Any]:
    ...

# ❌ Avoid
from typing import Union, Optional, Tuple

class MyDataset(Dataset):  # No type parameters
    def __getitem__(self, idx):  # No return type
        ...

def my_loader(self) -> DataLoader:  # No type parameter
    ...
```

### Docstrings
```python
def my_function(arg1: int, arg2: str) -> dict[str, Any]:
    """Brief description.

    Longer description explaining behavior in detail.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is invalid.

    Example:
        Usage example::

            result = my_function(42, "test")
    """
```

### Exception Handling
```python
# ✅ Correct - use logging.exception in except blocks
except Exception as exc:
    logger.exception(f"Error occurred: {exc}")
    raise RuntimeError(msg) from exc

# ❌ Avoid
except Exception as exc:
    logger.error(f"Error occurred: {exc}")
    raise RuntimeError(msg) from exc
```

### Import Ordering
```python
# ✅ Correct order
import logging  # stdlib
import shutil   # stdlib

from pathlib import Path  # stdlib

import torch  # third-party
from torch.utils.data import Dataset  # third-party

from deepsuite.lightning_base import BaseLoader  # local

# ❌ Avoid - mixed order
from deepsuite import BaseLoader
import torch
import logging
```

---

## 📊 Code Quality Dashboard

### Ruff Status
```
✅ 0 errors in core dataset modules
✅ All PEP 8 checks passing
✅ Import sorting validated
```

### MyPy Status
```
✅ 0 type errors in documented modules
✅ Generic types fully specified
✅ Function signatures complete
```

### Documentation Coverage
```
✅ 100% public API documented
✅ 100% of classes documented
✅ 100% of methods documented
✅ Examples provided for all modules
```

---

## 🚀 Common Tasks

### Add New Data Loading Class

1. **Inherit from BaseDataLoader**
```python
from deepsuite.lightning_base.dataset.base_loader import BaseDataLoader

class MyDataLoader(BaseDataLoader):
    """Your custom data loader."""
```

2. **Implement abstract methods**
```python
    def _get_train_transforms(self) -> Any:
        """Return training augmentation pipeline."""
        return MyAugmentations(...)

    def _get_val_transforms(self) -> Any:
        """Return validation preprocessing."""
        return MyValidation(...)

    def setup(self, stage: str | None = None) -> None:
        """Initialize train/val datasets."""
        self.train_dataset = MyDataset(...)
        self.val_dataset = MyDataset(...)
```

3. **Add type hints and docstrings**
```python
    def __init__(
        self,
        param1: int = 32,
        param2: str | None = None
    ) -> None:
        """Initialize with parameters.

        Args:
            param1: First parameter.
            param2: Optional second parameter.
        """
        super().__init__()
        self.param1 = param1
```

### Use Custom Tokenizer

```python
from deepsuite.lightning_base.dataset.text_loader import TextDataset

class MyTokenizer:
    def encode(self, text: str) -> list[int]:
        """Tokenize text to IDs."""
        return [...]

tokenizer = MyTokenizer()
dataset = TextDataset(
    data_path="data.txt",
    tokenizer=tokenizer,
    max_seq_len=512
)
```

---

## 📝 Configuration Examples

### ImageLoader Config
```python
# Conservative (good for transfer learning)
image_loader = ImageLoader(
    batch_size=32,
    num_workers=4,
    randaugment=False,  # Only minimal augmentation
    image_size=(224, 224)
)

# Aggressive (good for from-scratch training)
image_loader = ImageLoader(
    batch_size=64,
    num_workers=8,
    randaugment=True,   # Heavy augmentation
    image_size=(256, 256)
)
```

### AudioLoader Config
```python
# Minimal
audio_loader = AudioLoader(
    batch_size=32,
    num_workers=2,
    randaugment=False
)

# Aggressive
audio_loader = AudioLoader(
    batch_size=64,
    num_workers=8,
    randaugment=True
)
```

---

## 🔗 Related Files

- **Main Documentation**: `DOCUMENTATION.md`
- **Summary**: `DOCUMENTATION_SUMMARY.md`
- **Dataset Module**: `src/deepsuite/lightning_base/dataset/`
- **Base Classes**: `src/deepsuite/lightning_base/`

---

## ✨ Key Features

✅ **Auto-Download** - UniversalDataset handles downloads automatically
✅ **Type Safe** - Full type hints for IDE support and mypy
✅ **Well Documented** - Google-style docstrings with examples
✅ **Production Ready** - Passes ruff and mypy checks
✅ **Flexible** - Supports images, audio, text, and custom data
✅ **Lightning Integration** - Full PyTorch Lightning support

---

Need help? Check `DOCUMENTATION.md` for detailed API reference!
