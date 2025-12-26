"""Pytest configuration and fixtures."""

from pathlib import Path
import tempfile
from typing import Generator

import numpy as np
import pytest
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Create a sample image tensor."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def sample_audio() -> torch.Tensor:
    """Create a sample audio tensor."""
    return torch.randn(16000)  # 1 second at 16kHz


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a sample batch."""
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 10, (4,))
    return x, y


@pytest.fixture
def dummy_dataset_files(temp_dir: Path) -> Path:
    """Create dummy dataset files."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()

    # Create dummy images
    for i in range(10):
        img_path = data_dir / f"image_{i}.npy"
        np.save(img_path, np.random.rand(224, 224, 3))

    return data_dir


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
