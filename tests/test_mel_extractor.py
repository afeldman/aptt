"""Tests for MelSpectrogramExtractor."""

import numpy as np
import pytest
import torch

from deepsuite.layers.mel import MelSpectrogramExtractor


class TestMelSpectrogramExtractor:
    """Test suite for MelSpectrogramExtractor."""

    def test_initialization_default_params(self) -> None:
        """Test initialization with default parameters."""
        extractor = MelSpectrogramExtractor()

        assert extractor.sample_rate == 16000
        assert extractor.n_mels == 64
        assert extractor.n_fft == 400
        assert extractor.hop_length == 160
        assert extractor.f_min == 50
        assert extractor.f_max == 8000

    def test_initialization_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        extractor = MelSpectrogramExtractor(
            sample_rate=22050, n_mels=128, n_fft=512, hop_length=256, f_min=100, f_max=11000
        )

        assert extractor.sample_rate == 22050
        assert extractor.n_mels == 128
        assert extractor.n_fft == 512
        assert extractor.hop_length == 256
        assert extractor.f_min == 100
        assert extractor.f_max == 11000

    def test_mel_filterbank_creation(self) -> None:
        """Test that mel filterbank is created during initialization."""
        extractor = MelSpectrogramExtractor(n_mels=64, n_fft=400)

        assert hasattr(extractor, "mel_fb")
        assert extractor.mel_fb.shape == (64, 400 // 2 + 1)

    def test_forward_1d_input(self) -> None:
        """Test forward pass with 1D input."""
        extractor = MelSpectrogramExtractor(sample_rate=16000, n_mels=64, n_fft=400, hop_length=160)

        # 1 second of audio
        waveform = torch.randn(16000)
        output = extractor(waveform)

        # Should add batch dimension
        assert output.dim() == 3
        assert output.shape[0] == 1  # batch
        assert output.shape[1] == 64  # n_mels
        assert output.shape[2] > 0  # time frames

    def test_forward_2d_input(self) -> None:
        """Test forward pass with 2D input (batched)."""
        extractor = MelSpectrogramExtractor(sample_rate=16000, n_mels=64, n_fft=400, hop_length=160)

        # Batch of 4 audio samples
        waveform = torch.randn(4, 16000)
        output = extractor(waveform)

        assert output.shape[0] == 4  # batch
        assert output.shape[1] == 64  # n_mels
        assert output.shape[2] > 0  # time frames

    def test_output_shape_calculation(self) -> None:
        """Test output shape matches expected dimensions."""
        sample_rate = 16000
        n_mels = 80
        n_fft = 512
        hop_length = 256
        duration = 1.0  # 1 second

        extractor = MelSpectrogramExtractor(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )

        samples = int(sample_rate * duration)
        waveform = torch.randn(samples)
        output = extractor(waveform)

        expected_time_frames = (samples + hop_length - 1) // hop_length

        assert output.shape[1] == n_mels
        assert abs(output.shape[2] - expected_time_frames) <= 2  # Allow small diff

    def test_mel_scale_function(self) -> None:
        """Test the _mel_scale static method."""
        mel_fb = MelSpectrogramExtractor._mel_scale(
            sr=16000, n_fft=400, n_mels=64, f_min=50, f_max=8000
        )

        assert mel_fb.shape == (64, 400 // 2 + 1)
        assert isinstance(mel_fb, np.ndarray)
        # Check that filterbank is non-negative
        assert np.all(mel_fb >= 0)

    def test_mel_scale_properties(self) -> None:
        """Test properties of mel filterbank."""
        mel_fb = MelSpectrogramExtractor._mel_scale(
            sr=16000, n_fft=400, n_mels=64, f_min=50, f_max=8000
        )

        # Each mel filter should have non-zero values
        assert np.all(mel_fb.sum(axis=1) > 0)

        # Filterbank should be sparse (most values are zero)
        sparsity = (mel_fb == 0).sum() / mel_fb.size
        assert sparsity > 0.5  # At least 50% sparse

    def test_log_mel_spectrogram(self) -> None:
        """Test that output is in log scale."""
        extractor = MelSpectrogramExtractor()
        waveform = torch.randn(16000)
        output = extractor(waveform)

        # Log values should be negative for small magnitudes
        assert torch.any(output < 0)

    def test_different_sample_rates(self) -> None:
        """Test with different sample rates."""
        for sr in [8000, 16000, 22050, 44100]:
            extractor = MelSpectrogramExtractor(sample_rate=sr)
            waveform = torch.randn(sr)  # 1 second
            output = extractor(waveform)

            assert output.shape[1] == 64  # n_mels
            assert output.shape[2] > 0  # time frames

    def test_different_n_mels(self) -> None:
        """Test with different number of mel bands."""
        for n_mels in [40, 64, 80, 128]:
            extractor = MelSpectrogramExtractor(n_mels=n_mels)
            waveform = torch.randn(16000)
            output = extractor(waveform)

            assert output.shape[1] == n_mels

    def test_short_audio(self) -> None:
        """Test with very short audio."""
        extractor = MelSpectrogramExtractor(n_fft=400, hop_length=160)
        waveform = torch.randn(500)  # Less than n_fft
        output = extractor(waveform)

        # Should still produce output
        assert output.shape[2] > 0

    def test_long_audio(self) -> None:
        """Test with long audio."""
        extractor = MelSpectrogramExtractor()
        waveform = torch.randn(160000)  # 10 seconds
        output = extractor(waveform)

        assert output.shape[2] > 100  # Many time frames

    def test_zero_audio(self) -> None:
        """Test with zero (silent) audio."""
        extractor = MelSpectrogramExtractor()
        waveform = torch.zeros(16000)
        output = extractor(waveform)

        # Log of near-zero should be very negative
        assert torch.all(output < -10)

    def test_batch_processing_consistency(self) -> None:
        """Test that batched and single processing give same results."""
        extractor = MelSpectrogramExtractor()

        # Single sample
        waveform1 = torch.randn(16000)
        output1 = extractor(waveform1)

        # Same sample in batch
        waveform_batch = waveform1.unsqueeze(0)
        output_batch = extractor(waveform_batch)

        assert torch.allclose(output1[0], output_batch[0], rtol=1e-5)

    def test_device_compatibility(self) -> None:
        """Test that extractor works on different devices."""
        extractor = MelSpectrogramExtractor()
        waveform = torch.randn(16000)

        # CPU
        output_cpu = extractor(waveform)
        assert output_cpu.device.type == "cpu"

        # MPS/CUDA if available
        if torch.cuda.is_available():
            extractor_cuda = extractor.cuda()
            waveform_cuda = waveform.cuda()
            output_cuda = extractor_cuda(waveform_cuda)
            assert output_cuda.device.type == "cuda"
        elif torch.backends.mps.is_available():
            extractor_mps = extractor.to("mps")
            waveform_mps = waveform.to("mps")
            output_mps = extractor_mps(waveform_mps)
            assert output_mps.device.type == "mps"

    def test_mel_fb_buffer_registered(self) -> None:
        """Test that mel_fb is registered as a buffer."""
        extractor = MelSpectrogramExtractor()

        # Check that mel_fb is in buffers
        buffer_names = [name for name, _ in extractor.named_buffers()]
        assert "mel_fb" in buffer_names

    def test_frequency_range(self) -> None:
        """Test that f_min and f_max are respected."""
        extractor = MelSpectrogramExtractor(f_min=200, f_max=4000)

        # The mel filterbank should reflect the frequency range
        assert extractor.f_min == 200
        assert extractor.f_max == 4000
