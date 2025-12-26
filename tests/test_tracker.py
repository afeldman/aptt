"""Tests for Tracker module."""

import pytest
import torch

from deepsuite.model.tracking.tracker import Tracker


class DummyDetectionModel(torch.nn.Module):
    """Dummy detection model for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dummy bounding boxes."""
        batch_size = x.shape[0]
        # Return [batch, num_boxes, 4] (x1, y1, x2, y2)
        return torch.randn(batch_size, 5, 4)


class DummyReIDEncoder(torch.nn.Module):
    """Dummy ReID encoder for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dummy features."""
        batch_size = x.shape[0]
        return torch.randn(batch_size, 128)


class TestTracker:
    """Test suite for Tracker."""

    def test_initialization_without_reid(self) -> None:
        """Test initialization without ReID encoder."""
        detection_model = DummyDetectionModel()
        tracker = Tracker(detection_model=detection_model)

        assert tracker.detection_model is detection_model
        assert tracker.reid_encoder is None
        assert tracker.tracker is not None

    def test_initialization_with_reid(self) -> None:
        """Test initialization with ReID encoder."""
        detection_model = DummyDetectionModel()
        reid_encoder = DummyReIDEncoder()

        tracker = Tracker(detection_model=detection_model, reid_encoder=reid_encoder)

        assert tracker.detection_model is detection_model
        assert tracker.reid_encoder is reid_encoder

    def test_initialization_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        detection_model = DummyDetectionModel()

        tracker = Tracker(detection_model=detection_model, hidden_dim=256, rnn_type="LSTM")

        assert tracker.tracker is not None

    def test_forward_single_frame(self) -> None:
        """Test forward pass with single frame."""
        detection_model = DummyDetectionModel()
        tracker = Tracker(detection_model=detection_model)

        # Single frame
        frame = torch.randn(1, 3, 224, 224)
        frames = [frame]

        # Should not raise error
        output = tracker(frames)
        assert output is not None

    def test_forward_multiple_frames(self) -> None:
        """Test forward pass with multiple frames."""
        detection_model = DummyDetectionModel()
        tracker = Tracker(detection_model=detection_model)

        # Multiple frames
        frames = [torch.randn(1, 3, 224, 224) for _ in range(5)]

        output = tracker(frames)
        assert output is not None

    def test_forward_with_reid(self) -> None:
        """Test forward pass with ReID encoder."""
        detection_model = DummyDetectionModel()
        reid_encoder = DummyReIDEncoder()

        tracker = Tracker(detection_model=detection_model, reid_encoder=reid_encoder)

        frames = [torch.randn(1, 3, 224, 224) for _ in range(3)]
        output = tracker(frames)
        assert output is not None

    def test_crop_boxes_method(self) -> None:
        """Test crop_boxes method."""
        detection_model = DummyDetectionModel()
        tracker = Tracker(detection_model=detection_model)

        frame = torch.randn(3, 224, 224)
        boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float32)

        # Should handle box cropping
        crops = tracker.crop_boxes(frame, boxes)
        assert crops is not None

    def test_device_compatibility(self) -> None:
        """Test tracker on different devices."""
        detection_model = DummyDetectionModel()
        tracker = Tracker(detection_model=detection_model)

        # CPU
        frames = [torch.randn(1, 3, 224, 224)]
        output_cpu = tracker(frames)
        assert output_cpu is not None

    def test_lstm_rnn_type(self) -> None:
        """Test with LSTM RNN type."""
        detection_model = DummyDetectionModel()
        tracker = Tracker(detection_model=detection_model, rnn_type="LSTM")

        assert tracker.tracker is not None

    def test_gru_rnn_type(self) -> None:
        """Test with GRU RNN type."""
        detection_model = DummyDetectionModel()
        tracker = Tracker(detection_model=detection_model, rnn_type="GRU")

        assert tracker.tracker is not None

    def test_empty_frames_list(self) -> None:
        """Test with empty frames list."""
        detection_model = DummyDetectionModel()
        tracker = Tracker(detection_model=detection_model)

        frames = []
        # Should handle empty list gracefully
        output = tracker(frames)
        assert output is not None
