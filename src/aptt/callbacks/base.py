from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger

from aptt.utils.device import get_best_device


class ExportBaseCallback(pl.callbacks.ModelCheckpoint):
    """Base export callback that automatically fetches a batch from the trainer."""

    def __init__(self, output_dir="models", **kwargs):
        """Initialize the ExportBaseCallback.

        Args:
            output_dir (str): Directory where model exports should be stored.
            **kwargs: Additional arguments passed to the parent ModelCheckpoint.
        """
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.example_input = None  # Will be set automatically later

    def get_example_input(self, trainer: pl.Trainer):
        """Fetch a batch from the datamodule and save it as `self.example_input`.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer.

        Returns:
            torch.Tensor | Any: Example input batch.

        Example:
            ```python
            from pytorch_lightning import Trainer
            from aptt.checkpoint import ExportBaseCallback

            trainer = Trainer(callbacks=[ExportBaseCallback()])
            trainer.fit(model)
            ```
        """
        if self.example_input is None:  # Only fetch once
            try:
                datamodule = trainer.datamodule
                if datamodule is None:
                    logger.error("❌ No DataModule found.")
                    return None

                dataloader = datamodule.train_dataloader()
                if dataloader is None:
                    logger.error("❌ No train_dataloader() found.")
                    return None

                batch = next(iter(dataloader))  # Take one batch

                if isinstance(batch, tuple | list):
                    self.example_input = batch[0]
                else:
                    self.example_input = batch

                # Move to same device as model
                if trainer.model is not None:
                    device = trainer.model.device
                    self.example_input = self._move_batch_to_device(
                        self.example_input, device
                    )

                logger.info("✅ Example batch automatically loaded and moved to the correct device.")

            except Exception as e:
                logger.error(f"❌ Could not load example batch: {e}")
                return None

        return self.example_input

    def _move_batch_to_device(self, batch: Any, device: str | None = None):
        """Recursively move a batch to the specified device.

        Args:
            batch (Any): A tensor or (nested) list/tuple/dict of tensors.
            device (str): Target device (e.g., 'cuda' or 'cpu').

        Returns:
            Any: Batch moved to the target device.
        """
        if device is None:
            device = get_best_device()

        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, tuple | list):
            return type(batch)(self._move_batch_to_device(b, device) for b in batch)
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        return batch
