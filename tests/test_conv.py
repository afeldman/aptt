"""Unit tests for Conv modules."""

import pytest
import torch

from deepsuite.model.conv import BaseConv2dBlock, CausalConv1d, DepthwiseSeparableConv, SEBlock


class TestBaseConv2dBlock:
    """Test suite for BaseConv2dBlock."""

    def test_initialization_default(self) -> None:
        """Test initialization with default parameters."""
        block = BaseConv2dBlock(in_channels=3, out_channels=64)
        assert block.conv.in_channels == 3
        assert block.conv.out_channels == 64

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        block = BaseConv2dBlock(in_channels=3, out_channels=64, kernel_size=3)
        x = torch.randn(2, 3, 32, 32)
        output = block(x)
        assert output.shape == (2, 64, 32, 32)

    def test_with_stride(self) -> None:
        """Test with stride reduction."""
        block = BaseConv2dBlock(in_channels=3, out_channels=64, stride=2)
        x = torch.randn(2, 3, 32, 32)
        output = block(x)
        assert output.shape == (2, 64, 16, 16)


class TestDepthwiseSeparableConv:
    """Test suite for DepthwiseSeparableConv."""

    def test_initialization(self) -> None:
        """Test initialization."""
        conv = DepthwiseSeparableConv(in_channels=32, out_channels=64)
        assert isinstance(conv, torch.nn.Module)

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        conv = DepthwiseSeparableConv(in_channels=32, out_channels=64, kernel_size=3)
        x = torch.randn(2, 32, 16, 16)
        output = conv(x)
        assert output.shape == (2, 64, 16, 16)

    def test_with_stride(self) -> None:
        """Test with stride reduction."""
        conv = DepthwiseSeparableConv(in_channels=32, out_channels=64, stride=2)
        x = torch.randn(2, 32, 32, 32)
        output = conv(x)
        assert output.shape == (2, 64, 16, 16)


class TestSEBlock:
    """Test suite for SEBlock (Squeeze-and-Excitation)."""

    def test_initialization(self) -> None:
        """Test initialization."""
        se = SEBlock(channels=64, reduction=16)
        assert isinstance(se, torch.nn.Module)

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        se = SEBlock(channels=64)
        x = torch.randn(2, 64, 16, 16)
        output = se(x)
        assert output.shape == x.shape  # Should preserve shape

    def test_squeeze_excitation_applied(self) -> None:
        """Test that SE block actually modulates the input."""
        se = SEBlock(channels=64)
        x = torch.randn(2, 64, 16, 16)
        output = se(x)

        # Output should be different (recalibrated)
        assert not torch.equal(x, output)


class TestCausalConv1d:
    """Test suite for CausalConv1d."""

    def test_initialization(self) -> None:
        """Test initialization."""
        conv = CausalConv1d(in_channels=16, out_channels=32, kernel_size=3)
        assert isinstance(conv, torch.nn.Module)

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        conv = CausalConv1d(in_channels=16, out_channels=32, kernel_size=3)
        x = torch.randn(2, 16, 100)  # [B, C, T]
        output = conv(x)
        assert output.shape == (2, 32, 100)  # Should preserve temporal dimension

    def test_causality(self) -> None:
        """Test that future information does not leak."""
        conv = CausalConv1d(in_channels=16, out_channels=32, kernel_size=5)
        x = torch.randn(1, 16, 100)

        # Modify future timesteps
        x_modified = x.clone()
        x_modified[:, :, 50:] = 0

        output = conv(x)
        output_modified = conv(x_modified)

        # First 50 timesteps should be identical (causal)
        assert torch.allclose(output[:, :, :49], output_modified[:, :, :49], atol=1e-5)
