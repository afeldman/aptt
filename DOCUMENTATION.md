# DeepSuite Documentation

## Code Quality & Style

This project uses:
- **Google Style Docstrings** in English for all public APIs
- **Type Hints** (PEP 604) compatible with Python 3.11+
- **Ruff** for code linting and formatting
- **MyPy** for static type checking

## Dataset Module (`src/deepsuite/lightning_base/dataset/`)

### Core Components

#### UniversalDataset
PyTorch Dataset with optional auto-download and archive extraction.

**Features:**
- Automatic dataset downloading from URLs
- Support for multiple archive formats (.zip, .tar.gz, .tar, .tgz)
- Intelligent caching to avoid re-downloading
- Full type hints for IDE support

**Supported Archive Formats:**
- `.zip` - ZIP archives
- `.tar.gz`, `.tgz` - Gzipped TAR archives
- `.tar` - TAR archives

**Example Usage:**
```python
from deepsuite.lightning_base.dataset.universal_set import UniversalDataset

# Basic usage with auto-download
dataset = UniversalDataset(
    download_url="https://example.com/dataset.zip",
    root_dir="./data",
    auto_download=True
)

# With custom data and transforms
import torch
from torchvision import transforms

data = [torch.randn(3, 224, 224) for _ in range(100)]
labels = [i % 10 for i in range(100)]
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = UniversalDataset(
    data=data,
    labels=labels,
    transform=transform
)
```

---

#### BaseDataLoader
Abstract base class for PyTorch Lightning data modules.

**Key Methods:**
- `_get_train_transforms()` - Return training augmentation pipeline
- `_get_val_transforms()` - Return validation preprocessing pipeline
- `setup()` - Initialize train/val datasets
- `train_dataloader()` - Return training DataLoader
- `val_dataloader()` - Return validation DataLoader

**Type Annotations:**
```python
def train_dataloader(self) -> DataLoader[Any]:
    """Returns DataLoader for training with shuffling enabled."""

def val_dataloader(self) -> DataLoader[Any]:
    """Returns DataLoader for validation with shuffling disabled."""
```

---

#### ImageLoader
PyTorch Lightning data module for image datasets.

**Features:**
- Albumentations-based augmentation pipelines
- Standard and RandAugment-style aggressive augmentation
- ImageNet normalization by default
- Support for custom image sizes

**Augmentation Strategies:**

*Standard (randaugment=False):*
- Horizontal flip (50%)
- Random brightness/contrast (20%)
- ImageNet normalization

*RandAugment (randaugment=True):*
- Geometric: shift, scale, rotation, horizontal flip
- Texture: coarse dropout (0-1 holes, 5-32px)
- Color: brightness/contrast, hue/saturation, RGB shift
- Noise: Gaussian noise

**Example:**
```python
from deepsuite.lightning_base.dataset.image_loader import ImageLoader
from pytorch_lightning import Trainer

datamodule = ImageLoader(
    batch_size=32,
    num_workers=4,
    randaugment=True,
    image_size=(224, 224)
)

datamodule.setup()
trainer = Trainer(max_epochs=10)
trainer.fit(model, datamodule)
```

---

#### AudioLoader
PyTorch Lightning data module for audio datasets.

**Features:**
- Audiomentations-based augmentation pipelines
- Support for various audio signal processing techniques
- Normalization applied to all pipelines

**Augmentation Strategies:**

*Standard (randaugment=False):*
- Gaussian noise (1-15mV, 50%)
- Normalization

*RandAugment (randaugment=True):*
- Gaussian noise injection (1-15mV)
- Time stretching (0.8x-1.25x)
- Pitch shifting (±4 semitones)
- Clipping distortion (0-20% threshold)
- Gain adjustment (±12dB)
- Random shifts (±50% signal length)
- Normalization

**Example:**
```python
from deepsuite.lightning_base.dataset.audio_loader import AudioLoader

datamodule = AudioLoader(
    batch_size=64,
    num_workers=8,
    randaugment=True
)

datamodule.setup()
```

---

#### TextDataset & TextDataLoader
Dataset and DataModule for language modeling tasks.

**Features:**
- Multiple input format support (.txt, .jsonl, .pt)
- Flexible tokenization with multiple tokenizer backends
- Sliding window sampling for long sequences
- Multi-Token Prediction (MTP) support for improved LM training

**Supported Tokenizers:**
- HuggingFace transformers
- HuggingFace tokenizers library
- Custom tokenizers with `encode()` method

**Example:**
```python
from transformers import AutoTokenizer
from deepsuite.lightning_base.dataset.text_loader import TextDataset

tokenizer = AutoTokenizer.from_pretrained("gpt2")

dataset = TextDataset(
    data_path="data/corpus.txt",
    tokenizer=tokenizer,
    max_seq_len=512,
    stride=256,
    return_mtp=True,
    mtp_depth=4
)

# Access samples
sample = dataset[0]
print(sample["input_ids"].shape)  # torch.Size([512])
```

---

## Code Quality Checks

### Running Ruff (Linting & Formatting)
```bash
# Check for issues
uv run ruff check src/deepsuite/lightning_base/dataset/

# Auto-fix issues (including unsafe fixes)
uv run ruff check src/deepsuite/lightning_base/dataset/ --fix --unsafe-fixes

# View help
uv run ruff check --help
```

### Running MyPy (Type Checking)
```bash
# Type check specific modules
uv run mypy src/deepsuite/lightning_base/dataset/ --ignore-missing-imports

# With verbose output
uv run mypy src/deepsuite/lightning_base/dataset/ --ignore-missing-imports --show-error-codes
```

### Ruff Rules Applied
- **UP045**: PEP 604 union syntax (`X | Y` instead of `Union[X, Y]`)
- **I001**: Import sorting
- **F401**: Unused imports
- **W293**: Blank line whitespace
- **E/W**: PEP 8 style issues
- **N812**: Naming conventions (with `# noqa: N812` for approved exceptions like `A` for `albumentations`)
- **S310**: URL security audits (with `# noqa: S310` for intentional URL downloads)
- **TRY400**: Use `logging.exception()` instead of `logging.error()`
- **TRY301**: Abstract raises to inner functions (for better exception handling)
- **SIM108**: Ternary operators instead of simple if-else blocks

### Type Hints Conventions

**Dataset Type Annotations:**
```python
# Generic Dataset with tuple output
class MyDataset(Dataset[tuple[Any, Any]]):
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        ...

# DataLoader with generic type
def train_dataloader(self) -> DataLoader[Any]:
    ...

# Optional types
train_dataset: Dataset[Any] | None = None
```

**Union Types (PEP 604):**
```python
# Instead of Union[X, Y]
def process(value: str | int) -> dict[str, Any]:
    ...

# Optional is still X | None
def setup(self, stage: str | None = None) -> None:
    ...
```

---

## Documentation Style Guide

All public APIs use Google-style docstrings with the following structure:

```python
def method(arg1: int, arg2: str) -> dict[str, Any]:
    """Brief one-line description.

    Longer description explaining the method's purpose and behavior.
    Can span multiple lines and include implementation details.

    Args:
        arg1: Description of arg1. Type hints are in signature.
        arg2: Description of arg2 with expected values.

    Returns:
        Description of return value and its structure.

    Raises:
        ValueError: When something is invalid.
        RuntimeError: When runtime issue occurs.

    Example:
        Basic usage example::

            result = method(42, "test")
            print(result)  # output
    """
```

### Docstring Sections:
1. **Brief description** (one line)
2. **Detailed description** (optional)
3. **Args** - Parameter documentation
4. **Returns** - Return value documentation
5. **Raises** - Exception documentation (if applicable)
6. **Example** - Code example with `::` prefix for code blocks
7. **Note** (optional) - Important information

---

## Integration with CI/CD

To integrate these checks in your CI pipeline:

```yaml
# .github/workflows/lint.yml (example)
name: Lint & Type Check

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: astral-sh/setup-uv@v1
      - name: Ruff Check
        run: uv run ruff check src/
      - name: MyPy Check
        run: uv run mypy src/
```

---

## Dataset Module Summary

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `universal_set.py` | Auto-download & extract datasets | `UniversalDataset` |
| `base_loader.py` | Abstract Lightning DataModule | `BaseDataLoader` |
| `image_loader.py` | Image data loading with augmentation | `ImageLoader` |
| `audio_loader.py` | Audio data loading with augmentation | `AudioLoader` |
| `text_loader.py` | Text/LLM data loading | `TextDataset`, `TextDataLoader` |

All modules are fully documented with type hints and Google-style docstrings.
