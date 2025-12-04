# APTT Coding Standards

> Code Quality Standards für APTT v0.2.0

## 🎯 Übersicht

APTT verwendet moderne Python-Tooling für Code-Qualität:

- **Ruff**: Linting und Formatting (ersetzt Flake8, isort, pyupgrade)
- **MyPy**: Static Type Checking
- **Google-Style Docstrings**: Konsistente Dokumentation

## 📝 Docstring Style Guide

### Google-Style Docstring Format

Alle öffentlichen Module, Klassen, Methoden und Funktionen müssen Docstrings im Google-Style haben.

#### Module Docstring

```python
"""Multi-Head Latent Attention Implementation.

This module implements MLA with Low-Rank KV-Compression for efficient
KV-Cache during inference.

Reference:
    DeepSeek-V3 Technical Report - https://arxiv.org/html/2412.19437v2
"""
```

#### Class Docstring

```python
class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention mit Low-Rank KV-Compression.

    MLA komprimiert Keys und Values gemeinsam in einen niedrigdimensionalen
    Latent Space, was den KV-Cache während der Inferenz drastisch reduziert.

    Args:
        d: Embedding dimension (e.g., 7168 for DeepSeek-V3).
        n_h: Number of attention heads (e.g., 128).
        d_h: Dimension per head. Defaults to d // n_h.
        d_c: KV compression dimension, must be << d (e.g., 512).
        d_c_q: Query compression dimension (e.g., 1536).
        d_h_R: Per-head RoPE dimension (e.g., 64).
        dropout: Dropout probability for attention weights. Defaults to 0.0.

    Attributes:
        scale: Scale factor for attention scores (d_h ** -0.5).
        W_D_Q: Query down-projection layer.
        W_U_Q: Query up-projection layer.
        rope_q: Rotary position embedding for queries.
        kv_compression: KV compression module.
        W_O: Output projection layer.

    Shape:
        - Input: (batch_size, seq_len, d)
        - Output: (batch_size, seq_len, d)

    Memory Savings:
        Standard MHA KV-Cache: 2 * seq_len * n_h * d_h
        MLA KV-Cache: seq_len * d_c
        Reduction: ~32x with DeepSeek-V3 parameters

    Examples:
        >>> # Initialize MLA with DeepSeek-V3 config
        >>> mla = MultiHeadLatentAttention(
        ...     d=7168, n_h=128, d_h=128,
        ...     d_c=512, d_c_q=1536, d_h_R=64
        ... )
        >>>
        >>> # Forward pass
        >>> x = torch.randn(2, 512, 7168)  # (B, L, D)
        >>> output, cache = mla(x, use_cache=True)
        >>> print(output.shape)  # (2, 512, 7168)
        >>>
        >>> # Use cached KV for next token
        >>> x_next = torch.randn(2, 1, 7168)
        >>> output_next, cache = mla(x_next, use_cache=True, past_key_value=cache)

    Note:
        This implementation uses Flash Attention when available (PyTorch >= 2.0)
        for improved performance.

    References:
        Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
        DeepSeek-AI (2024). DeepSeek-V3 Technical Report.
    """
```

#### Method/Function Docstring

```python
def forward(
    self,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    use_cache: bool = False,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Forward pass through Multi-Head Latent Attention.

    Computes attention with optional KV-caching for efficient autoregressive
    generation.

    Args:
        x: Input tensor of shape (batch_size, seq_len, d).
        attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            or (batch_size, 1, seq_len, seq_len). Values should be 0 (mask) or 1 (keep).
            Defaults to None (causal masking).
        use_cache: Whether to return cached keys/values for next step. Used during
            autoregressive generation. Defaults to False.
        past_key_value: Cached (keys, values) from previous step, each of shape
            (batch_size, prev_seq_len, n_h, d_h). Only used when use_cache=True.
            Defaults to None.

    Returns:
        A tuple containing:
            - output: Attention output of shape (batch_size, seq_len, d).
            - cached_kv: Optional tuple of (keys, values) for caching, each of shape
              (batch_size, total_seq_len, n_h, d_h). None if use_cache=False.

    Raises:
        ValueError: If past_key_value is provided but use_cache is False.
        RuntimeError: If attention_mask shape is incompatible with input.

    Examples:
        >>> mla = MultiHeadLatentAttention(d=512, n_h=8, d_h=64, d_c=128)
        >>>
        >>> # Training: single forward pass
        >>> x = torch.randn(2, 100, 512)
        >>> output, _ = mla(x)
        >>>
        >>> # Inference: autoregressive with caching
        >>> x_prompt = torch.randn(1, 50, 512)
        >>> output, cache = mla(x_prompt, use_cache=True)
        >>>
        >>> # Generate next token
        >>> x_next = torch.randn(1, 1, 512)
        >>> output_next, cache = mla(x_next, use_cache=True, past_key_value=cache)

    Note:
        When use_cache=True, keys and values are accumulated across calls.
        The cached tensors grow in the seq_len dimension.
    """
```

#### Property Docstring

```python
@property
def compression_ratio(self) -> float:
    """Compute KV-cache compression ratio.

    Returns:
        Compression ratio as a float (e.g., 32.0 means 32x smaller cache).

    Examples:
        >>> mla = MultiHeadLatentAttention(d=7168, n_h=128, d_h=128, d_c=512)
        >>> print(mla.compression_ratio)
        32.0
    """
    return (2 * self.n_h * self.d_h) / self.d_c
```

#### Simple Function Docstring

```python
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal attention mask for autoregressive modeling.

    Args:
        seq_len: Sequence length.
        device: Device to create mask on.

    Returns:
        Lower triangular mask of shape (seq_len, seq_len).

    Examples:
        >>> mask = create_causal_mask(4, torch.device("cpu"))
        >>> print(mask)
        tensor([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 1., 1.]])
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device))
```

### Docstring Sections (in Order)

1. **Summary**: One-line description (mandatory)
2. **Extended Description**: Detailed explanation (optional)
3. **Args**: Function/method arguments (if applicable)
4. **Attributes**: Class attributes (for classes only)
5. **Returns**: Return value(s) (if applicable)
6. **Yields**: For generators (if applicable)
7. **Raises**: Exceptions that can be raised (if applicable)
8. **Warns**: Warnings that can be issued (if applicable)
9. **Shape**: Tensor shapes for PyTorch modules (recommended)
10. **Examples**: Usage examples with doctest format (highly recommended)
11. **Note**: Additional notes (optional)
12. **See Also**: Related functions/classes (optional)
13. **References**: Citations (optional)

## 🔍 Type Annotations

### General Rules

- **All functions/methods** müssen Type Annotations haben
- Nutze `from __future__ import annotations` für forward references
- Nutze `typing` module für komplexe Types
- Nutze `|` statt `Union` (Python 3.10+)
- Nutze `list[T]` statt `List[T]` (Python 3.9+)

### Examples

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

T = TypeVar("T", bound=nn.Module)


def create_optimizer(
    model: nn.Module,
    optimizer_type: Literal["adam", "adamw", "sgd"] = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    **kwargs: dict[str, float],
) -> torch.optim.Optimizer:
    """Create optimizer for model training.

    Args:
        model: PyTorch model to optimize.
        optimizer_type: Type of optimizer to use. Must be one of "adam", "adamw", "sgd".
        learning_rate: Learning rate for optimizer.
        weight_decay: L2 regularization coefficient.
        betas: Betas for Adam-based optimizers.
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer_type is not supported.

    Examples:
        >>> model = nn.Linear(10, 5)
        >>> optimizer = create_optimizer(model, "adamw", learning_rate=1e-3)
        >>> print(type(optimizer))
        <class 'torch.optim.adamw.AdamW'>
    """
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            **kwargs,
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            **kwargs,
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        msg = f"Unsupported optimizer type: {optimizer_type}"
        raise ValueError(msg)


def process_batch(
    batch: dict[str, torch.Tensor],
    transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    """Process a batch of data with optional transforms.

    Args:
        batch: Dictionary mapping keys to tensors.
        transforms: Optional sequence of transform functions to apply.

    Returns:
        Processed batch dictionary.

    Examples:
        >>> batch = {"image": torch.randn(2, 3, 224, 224), "label": torch.tensor([0, 1])}
        >>> processed = process_batch(batch)
    """
    if transforms is None:
        return batch

    result = {}
    for key, tensor in batch.items():
        result[key] = tensor
        for transform in transforms:
            result[key] = transform(result[key])

    return result
```

### PyTorch-Specific Types

```python
import torch
import torch.nn as nn
from torch import Tensor


class CustomLayer(nn.Module):
    """Custom layer with proper type annotations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize custom layer.

        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            bias: Whether to include bias term.
            device: Device to create parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        return nn.functional.linear(x, self.weight, self.bias)
```

## 🛠️ Development Workflow

### 1. Before Committing

Führe immer diese Checks aus:

```bash
# Format code with Ruff
ruff format .

# Fix auto-fixable issues
ruff check --fix .

# Check remaining issues
ruff check .

# Type check with MyPy
mypy aptt/
```

### 2. Configuration Files

Alle Konfigurationen sind in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "D", "UP", "ANN", ...]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.mypy]
python_version = "3.11"
strict = true
```

### 3. Pre-commit Hook (Optional)

Erstelle `.git/hooks/pre-commit`:

```bash
#!/bin/bash
set -e

echo "Running Ruff format..."
ruff format .

echo "Running Ruff checks..."
ruff check --fix .

echo "Running MyPy..."
mypy aptt/

echo "All checks passed!"
```

Mache es ausführbar:

```bash
chmod +x .git/hooks/pre-commit
```

## ✅ Code Quality Checklist

Vor jedem Commit:

- [ ] Alle öffentlichen APIs haben Google-Style Docstrings
- [ ] Alle Funktionen/Methoden haben Type Annotations
- [ ] Docstrings enthalten Examples mit doctest format
- [ ] `ruff format .` wurde ausgeführt
- [ ] `ruff check .` zeigt keine Fehler
- [ ] `mypy aptt/` zeigt keine Fehler
- [ ] Tensor shapes sind in Docstrings dokumentiert
- [ ] Komplexe Algorithmen haben Note/Reference Sections

## 📚 Resources

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)

## 🎯 Examples aus APTT

### Gutes Beispiel: MLA Module

```python
"""Multi-Head Latent Attention Implementation.

Referenz: DeepSeek-V3 Technical Report - https://arxiv.org/html/2412.19437v2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

from .kv_compression import KVCompression
from .rope import RotaryPositionEmbedding


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention mit Low-Rank KV-Compression.

    [Full docstring wie oben...]
    """

    def __init__(
        self,
        d: int,
        n_h: int,
        d_h: int | None = None,
        d_c: int = 512,
        d_c_q: int = 1536,
        d_h_R: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """Initialize MLA layer."""
        super().__init__()
        # Implementation...

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Forward pass through MLA."""
        # Implementation...
```

### Schlechtes Beispiel (vermeiden!)

```python
# ❌ Keine Docstrings
def forward(self, x, mask=None):
    return self.attn(x, mask)

# ❌ Keine Type Annotations
def process(data):
    return data.mean()

# ❌ Unklare Dokumentation
def compute(x):
    """Does stuff."""  # Zu vage!
    return x * 2
```

## 🔧 Troubleshooting

### MyPy Fehler: "Module not found"

Füge Module zu `[[tool.mypy.overrides]]` hinzu:

```toml
[[tool.mypy.overrides]]
module = ["your_module.*"]
ignore_missing_imports = true
```

### Ruff Fehler: "Line too long"

Nutze automatisches Formatting:

```bash
ruff format your_file.py
```

### Docstring Warnings

Häufige Ruff-Warnungen:

- `D102`: Missing docstring in public method → Füge Docstring hinzu
- `D103`: Missing docstring in public function → Füge Docstring hinzu
- `D417`: Missing argument description → Füge Args hinzu

---

**Version**: 0.2.0  
**Letzte Aktualisierung**: 4. Dezember 2025
