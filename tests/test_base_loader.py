"""Tests for BaseDataLoader."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from deepsuite.lightning_base.dataset.base_loader import BaseDataLoader


class DummyDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dummy dataset for testing."""

    def __init__(self, size: int = 100) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.randn(3, 32, 32), idx % 10


class ConcreteDataLoader(BaseDataLoader):
    """Concrete implementation of BaseDataLoader for testing."""

    def __init__(self, batch_size: int = 32, num_workers: int = 0) -> None:
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None

    def _get_train_transforms(self) -> Any:
        """Return identity transform."""
        return None

    def _get_val_transforms(self) -> Any:
        """Return identity transform."""
        return None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = DummyDataset(100)
            self.val_dataset = DummyDataset(20)


class TestBaseDataLoader:
    """Test suite for BaseDataLoader."""

    def test_initialization_default_params(self) -> None:
        """Test initialization with default parameters."""
        loader = ConcreteDataLoader()

        assert loader.batch_size == 32
        assert loader.num_workers == 0
        assert loader.randaugment is False

    def test_initialization_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        loader = ConcreteDataLoader(batch_size=64, num_workers=4)

        assert loader.batch_size == 64
        assert loader.num_workers == 4

    def test_setup_fit_stage(self) -> None:
        """Test setup method with fit stage."""
        loader = ConcreteDataLoader()
        loader.setup(stage="fit")

        assert loader.train_dataset is not None
        assert loader.val_dataset is not None
        assert len(loader.train_dataset) == 100
        assert len(loader.val_dataset) == 20

    def test_setup_none_stage(self) -> None:
        """Test setup method with None stage."""
        loader = ConcreteDataLoader()
        loader.setup(stage=None)

        assert loader.train_dataset is not None
        assert loader.val_dataset is not None

    def test_train_dataloader(self) -> None:
        """Test train_dataloader method."""
        loader = ConcreteDataLoader(batch_size=16, num_workers=0)
        loader.setup("fit")

        train_loader = loader.train_dataloader()

        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 16
        assert train_loader.num_workers == 0
        # Check shuffle is enabled for training
        assert hasattr(train_loader, "sampler")

    def test_val_dataloader(self) -> None:
        """Test val_dataloader method."""
        loader = ConcreteDataLoader(batch_size=32, num_workers=0)
        loader.setup("fit")

        val_loader = loader.val_dataloader()

        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 32
        assert val_loader.num_workers == 0

    def test_train_dataloader_before_setup_raises_error(self) -> None:
        """Test that train_dataloader raises error before setup."""
        loader = ConcreteDataLoader()

        with pytest.raises(RuntimeError, match="train_dataset is None"):
            loader.train_dataloader()

    def test_val_dataloader_before_setup_raises_error(self) -> None:
        """Test that val_dataloader raises error before setup."""
        loader = ConcreteDataLoader()

        with pytest.raises(RuntimeError, match="val_dataset is None"):
            loader.val_dataloader()

    def test_train_dataloader_iteration(self) -> None:
        """Test iterating through train dataloader."""
        loader = ConcreteDataLoader(batch_size=8, num_workers=0)
        loader.setup("fit")

        train_loader = loader.train_dataloader()
        batch = next(iter(train_loader))

        assert len(batch) == 2  # (x, y)
        assert batch[0].shape == (8, 3, 32, 32)
        assert batch[1].shape == (8,)

    def test_val_dataloader_iteration(self) -> None:
        """Test iterating through val dataloader."""
        loader = ConcreteDataLoader(batch_size=4, num_workers=0)
        loader.setup("fit")

        val_loader = loader.val_dataloader()
        batch = next(iter(val_loader))

        assert len(batch) == 2
        assert batch[0].shape == (4, 3, 32, 32)
        assert batch[1].shape == (4,)

    def test_multiple_setup_calls(self) -> None:
        """Test calling setup multiple times."""
        loader = ConcreteDataLoader()

        loader.setup("fit")
        first_train = loader.train_dataset

        loader.setup("fit")
        second_train = loader.train_dataset

        # Should create new datasets
        assert first_train is not second_train

    def test_num_workers_zero(self) -> None:
        """Test with num_workers=0 (single process)."""
        loader = ConcreteDataLoader(num_workers=0)
        loader.setup("fit")

        train_loader = loader.train_dataloader()
        assert train_loader.num_workers == 0

    def test_batch_size_one(self) -> None:
        """Test with batch_size=1."""
        loader = ConcreteDataLoader(batch_size=1)
        loader.setup("fit")

        train_loader = loader.train_dataloader()
        batch = next(iter(train_loader))

        assert batch[0].shape == (1, 3, 32, 32)
        assert batch[1].shape == (1,)

    def test_large_batch_size(self) -> None:
        """Test with large batch size."""
        loader = ConcreteDataLoader(batch_size=128)
        loader.setup("fit")

        train_loader = loader.train_dataloader()
        # Should not raise error
        batch = next(iter(train_loader))
        assert batch[0].shape[0] <= 128  # May be less if dataset is small
