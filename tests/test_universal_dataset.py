"""Tests for UniversalDataset."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import zipfile

import pytest
import torch

from deepsuite.lightning_base.dataset.universal_set import UniversalDataset


class TestUniversalDataset:
    """Test suite for UniversalDataset."""

    def test_basic_initialization(self, temp_dir: Path) -> None:
        """Test basic dataset initialization."""
        # Create dummy data
        data = [torch.randn(3, 32, 32) for _ in range(10)]
        labels = list(range(10))

        dataset = UniversalDataset(data=data, labels=labels)

        assert len(dataset) == 10
        assert dataset[0][0].shape == (3, 32, 32)
        assert dataset[0][1] == 0

    def test_with_transform(self, temp_dir: Path) -> None:
        """Test dataset with transform."""
        data = [torch.randn(3, 32, 32) for _ in range(5)]
        labels = list(range(5))

        def dummy_transform(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        dataset = UniversalDataset(data=data, labels=labels, transform=dummy_transform)

        original = data[0]
        transformed, label = dataset[0]

        assert torch.allclose(transformed, original * 2)
        assert label == 0

    def test_download_disabled_by_default(self, temp_dir: Path) -> None:
        """Test that download is disabled by default."""
        data = [torch.randn(3, 32, 32) for _ in range(3)]
        labels = [0, 1, 2]

        dataset = UniversalDataset(
            data=data,
            labels=labels,
            download_url="http://example.com/data.zip",
            root_dir=str(temp_dir),
        )

        # Should work without downloading
        assert len(dataset) == 3

    @patch("urllib.request.urlretrieve")
    def test_download_and_extract_zip(self, mock_urlretrieve: MagicMock, temp_dir: Path) -> None:
        """Test downloading and extracting a zip file."""
        # Create a dummy zip file
        zip_path = temp_dir / "data.zip"
        extract_dir = temp_dir / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.txt", "test data")

        # Mock urlretrieve to use our dummy zip
        mock_urlretrieve.return_value = (str(zip_path), None)

        data = [torch.randn(3, 32, 32) for _ in range(2)]
        labels = [0, 1]

        dataset = UniversalDataset(
            data=data,
            labels=labels,
            download_url="http://example.com/data.zip",
            root_dir=str(temp_dir),
            auto_download=True,
        )

        assert extract_dir.exists()
        assert (extract_dir / "data.txt").exists()

    @patch("urllib.request.urlretrieve")
    def test_download_and_extract_tar_gz(self, mock_urlretrieve: MagicMock, temp_dir: Path) -> None:
        """Test downloading and extracting a tar.gz file."""
        import tarfile

        # Create a dummy tar.gz file
        tar_path = temp_dir / "data.tar.gz"
        extract_dir = temp_dir / "extracted"

        with tarfile.open(tar_path, "w:gz") as tf:
            # Create a dummy file in memory
            import io

            dummy_file = io.BytesIO(b"test data")
            info = tarfile.TarInfo(name="data.txt")
            info.size = len(dummy_file.getvalue())
            dummy_file.seek(0)
            tf.addfile(info, dummy_file)

        # Mock urlretrieve
        mock_urlretrieve.return_value = (str(tar_path), None)

        data = [torch.randn(3, 32, 32) for _ in range(2)]
        labels = [0, 1]

        dataset = UniversalDataset(
            data=data,
            labels=labels,
            download_url="http://example.com/data.tar.gz",
            root_dir=str(temp_dir),
            auto_download=True,
        )

        assert extract_dir.exists()

    def test_unsupported_format_raises_error(self, temp_dir: Path) -> None:
        """Test that unsupported format raises ValueError."""
        data = [torch.randn(3, 32, 32) for _ in range(2)]
        labels = [0, 1]

        with pytest.raises(ValueError, match="Unsupported archive format"):
            dataset = UniversalDataset(
                data=data,
                labels=labels,
                download_url="http://example.com/data.rar",
                root_dir=str(temp_dir),
                auto_download=True,
            )

    def test_getitem_with_invalid_index(self, temp_dir: Path) -> None:
        """Test __getitem__ with invalid index."""
        data = [torch.randn(3, 32, 32) for _ in range(5)]
        labels = list(range(5))
        dataset = UniversalDataset(data=data, labels=labels)

        with pytest.raises(IndexError):
            _ = dataset[10]

    def test_len_method(self, temp_dir: Path) -> None:
        """Test __len__ method."""
        data = [torch.randn(3, 32, 32) for _ in range(7)]
        labels = list(range(7))
        dataset = UniversalDataset(data=data, labels=labels)

        assert len(dataset) == 7

    def test_empty_dataset(self) -> None:
        """Test empty dataset."""
        dataset = UniversalDataset(data=[], labels=[])

        assert len(dataset) == 0

        with pytest.raises(IndexError):
            _ = dataset[0]

    def test_no_download_when_extracted_exists(self, temp_dir: Path) -> None:
        """Test that download is skipped if extracted dir exists."""
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        (extract_dir / "marker.txt").write_text("already extracted")

        data = [torch.randn(3, 32, 32) for _ in range(2)]
        labels = [0, 1]

        with patch("urllib.request.urlretrieve") as mock_download:
            dataset = UniversalDataset(
                data=data,
                labels=labels,
                download_url="http://example.com/data.zip",
                root_dir=str(temp_dir),
                auto_download=True,
            )

            # Should not download if extracted dir exists
            mock_download.assert_not_called()

    def test_transform_none(self, temp_dir: Path) -> None:
        """Test dataset with transform=None."""
        data = [torch.randn(3, 32, 32) for _ in range(3)]
        labels = [0, 1, 2]
        dataset = UniversalDataset(data=data, labels=labels, transform=None)

        x, y = dataset[0]
        assert torch.equal(x, data[0])
        assert y == labels[0]
