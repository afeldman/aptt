"""Unit tests for Loss functions."""

import pytest
import torch

from deepsuite.loss.bbox import BboxLoss
from deepsuite.loss.classification import ClassificationLoss
from deepsuite.loss.focal import FocalLoss
from deepsuite.loss.varifocal import VarifocalLoss


class TestFocalLoss:
    """Test suite for FocalLoss."""

    def test_initialization(self) -> None:
        """Test initialization."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        assert isinstance(loss_fn, torch.nn.Module)

    def test_forward_binary(self) -> None:
        """Test forward pass for binary classification."""
        loss_fn = FocalLoss()
        logits = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()

        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_forward_multiclass(self) -> None:
        """Test forward pass for multiclass classification."""
        loss_fn = FocalLoss()
        logits = torch.randn(10, 5)  # 5 classes
        targets = torch.randint(0, 5, (10,))

        loss = loss_fn(logits, targets)
        assert loss >= 0


class TestVarifocalLoss:
    """Test suite for VarifocalLoss."""

    def test_initialization(self) -> None:
        """Test initialization."""
        loss_fn = VarifocalLoss()
        assert isinstance(loss_fn, torch.nn.Module)

    def test_forward(self) -> None:
        """Test forward pass."""
        loss_fn = VarifocalLoss()
        pred = torch.rand(10, 80)  # 80 classes
        target = torch.rand(10, 80)

        loss = loss_fn(pred, target)
        assert loss >= 0


class TestBboxLoss:
    """Test suite for BboxLoss."""

    def test_initialization(self) -> None:
        """Test initialization."""
        loss_fn = BboxLoss()
        assert isinstance(loss_fn, torch.nn.Module)

    def test_forward(self) -> None:
        """Test forward pass."""
        loss_fn = BboxLoss()
        pred_boxes = torch.rand(10, 4)  # 10 boxes, (x1, y1, x2, y2)
        target_boxes = torch.rand(10, 4)

        loss = loss_fn(pred_boxes, target_boxes)
        assert loss >= 0


class TestClassificationLoss:
    """Test suite for ClassificationLoss."""

    def test_initialization(self) -> None:
        """Test initialization."""
        loss_fn = ClassificationLoss()
        assert loss_fn is not None

    def test_forward(self) -> None:
        """Test forward pass."""
        loss_fn = ClassificationLoss()
        logits = torch.randn(10, 5)  # 10 samples, 5 classes
        targets = torch.randint(0, 5, (10,))

        loss = loss_fn(logits, targets)
        assert loss >= 0
