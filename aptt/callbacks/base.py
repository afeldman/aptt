from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger

from apu.ml.utils.device import get_best_device


class ExportBaseCallback(pl.callbacks.ModelCheckpoint):
    """
    Basis-Export-Callback, der automatisch einen Batch aus dem Trainer zieht.
    """

    def __init__(self, output_dir="models", **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.example_input = None  # Wird später automatisch gesetzt

    def get_example_input(self, trainer: pl.Trainer):
        """
        Holt sich einen Batch aus dem DataModule und speichert ihn als `self.example_input`.

        :param trainer: PyTorch Lightning Trainer.
        :return: Beispiel-Batch.

        :example:
        ```python
        from pytorch_lightning import Trainer
        from apu.ml.checkpoint import ExportBaseCallback

        trainer = Trainer(callbacks=[ExportBaseCallback()])
        trainer.fit(model)
        ```
        """
        if self.example_input is None:  # Nur einmal holen
            try:
                datamodule = trainer.datamodule
                if datamodule is None:
                    logger.error("❌ Kein DataModule gefunden.")
                    return None

                dataloader = datamodule.train_dataloader()
                if dataloader is None:
                    logger.error("❌ Kein train_dataloader() gefunden.")
                    return None

                batch = next(iter(dataloader))  # Nimm einen Batch

                if isinstance(batch, (tuple, list)):  # Falls (X, y)-Format
                    self.example_input = batch[0]
                else:
                    self.example_input = batch  # Falls nur X existiert

                # Gerät des Modells holen
                if trainer.model is not None:
                    device = trainer.model.device
                    self.example_input = self._move_batch_to_device(self.example_input, device)

                logger.info("✅ Beispiel-Batch automatisch geladen und auf das richtige Gerät verschoben.")

            except Exception as e:
                logger.error(f"❌ Konnte Beispiel-Batch nicht laden: {e}")
                return None

        return self.example_input

    def _move_batch_to_device(self, batch, device:str=get_best_device()):
        """
        Verschiebt einen Batch rekursiv auf das angegebene `device`.

        :param batch: Tensor oder (verschachtelte) Liste/Tuple/Dictionaries von Tensors.
        :param device: Zielgerät (z. B. 'cuda' oder 'cpu').
        :return: Batch auf dem Zielgerät.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, (tuple, list)):
            return type(batch)(self._move_batch_to_device(b, device) for b in batch)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        return batch
