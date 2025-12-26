"""Unit tests for Metrics."""

import pytest
import torch

from deepsuite.metric.bbox_iou import bbox_iou
from deepsuite.metric.norm import l2_norm


class TestBBoxIoU:
    """Test suite for bbox_iou function."""

    def test_perfect_overlap(self) -> None:
        """Test IoU for perfectly overlapping boxes."""
        box1 = torch.tensor([[10, 10, 50, 50]]).float()
        box2 = torch.tensor([[10, 10, 50, 50]]).float()

        iou = bbox_iou(box1, box2)
        assert torch.isclose(iou[0], torch.tensor(1.0))

    def test_no_overlap(self) -> None:
        """Test IoU for non-overlapping boxes."""
        box1 = torch.tensor([[10, 10, 20, 20]]).float()
        box2 = torch.tensor([[100, 100, 120, 120]]).float()

        iou = bbox_iou(box1, box2)
        assert torch.isclose(iou[0], torch.tensor(0.0))

    def test_partial_overlap(self) -> None:
        """Test IoU for partially overlapping boxes."""
        box1 = torch.tensor([[0, 0, 10, 10]]).float()
        box2 = torch.tensor([[5, 5, 15, 15]]).float()

        iou = bbox_iou(box1, box2)

        # Calculate expected IoU manually:
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25/175 = 0.1428...
        expected = 25 / 175
        assert torch.isclose(iou[0], torch.tensor(expected), atol=1e-4)

    def test_batch_iou(self) -> None:
        """Test batch IoU computation."""
        boxes1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]]).float()
        boxes2 = torch.tensor([[10, 10, 50, 50], [25, 25, 65, 65]]).float()

        iou = bbox_iou(boxes1, boxes2)

        assert iou.shape[0] == 2
        assert torch.isclose(iou[0], torch.tensor(1.0))  # Perfect overlap


class TestL2Norm:
    """Test suite for l2_norm."""

    def test_l2_norm_identical_vectors(self) -> None:
        """Test L2 norm of identical vectors."""
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[1.0, 2.0, 3.0]])

        norm = l2_norm(a, b)
        expected = torch.sqrt(torch.tensor([[14.0]]))  # 1*1 + 2*2 + 3*3 = 14
        assert torch.isclose(norm, expected, atol=1e-5)

    def test_l2_norm_orthogonal_vectors(self) -> None:
        """Test L2 norm of orthogonal vectors."""
        a = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 1.0, 0.0]])

        norm = l2_norm(a, b)
        # Dot product is 0, so norm should be sqrt(epsilon)
        assert norm < 0.01

    def test_l2_norm_batch(self) -> None:
        """Test L2 norm with batched input."""
        a = torch.rand(5, 10)
        b = torch.rand(5, 10)

        norm = l2_norm(a, b)
        assert norm.shape == (5, 1)
