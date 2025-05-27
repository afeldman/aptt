"""
TorchScript-Export-Callback für PyTorch Lightning.

Dieses Modul enthält einen Callback, der das beste Modell nach TorchScript exportiert, sobald es gespeichert wurde.
 
Beispiel:
```python
from pytorch_lightning import Trainer
from aptt.callbacks.torchscript import TorchScriptExportCallback
 
trainer = Trainer(callbacks=[TorchScriptExportCallback()])
trainer.fit(model)
```
"""

from pathlib import Path
 
import pytorch_lightning as pl
import torch
from loguru import logger
 
from aptt.callbacks.base import ExportBaseCallback
 
 
class TorchScriptExportCallback(ExportBaseCallback):
    """
    Exportiert das Modell nach TorchScript (.pt).
    """
 
    def __init__(self, output_dir="models", optimize=True, **kwargs):
        """
        :param output_dir: Verzeichnis für exportierte Modelle.
        :param optimize: Falls True, wird TorchScript optimiert.
        :param kwargs: Zusätzliche Parameter für ModelCheckpoint.
        """

        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimize = optimize
 
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Exportiert das Modell nach TorchScript (.pt).
 
        :param trainer: PyTorch Lightning Trainer.
        :param pl_module: PyTorch Lightning Modul.
 
        :example:
        ```python
        from pytorch_lightning import Trainer
        from aptt.checkpoint import TorchScriptExportCallback
 
        trainer = Trainer(callbacks=[TorchScriptExportCallback()])
        trainer.fit(model)
        ```
        """

        super().on_validation_end(trainer, pl_module)
 
        best_checkpoint_path = self.best_model_path
        if not best_checkpoint_path or not Path(best_checkpoint_path).exists():
            logger.warning("❌ Kein gültiger Checkpoint gefunden. Export übersprungen.")
            return
 
        logger.info(f"🔹 Lade bestes Modell: {best_checkpoint_path}")
 
        pl_module.load_state_dict(torch.load(best_checkpoint_path)["state_dict"])
        pl_module.eval()
 
        # Beispiel-Input automatisch holen
        example_input = self.get_example_input(trainer)
        if example_input is None:
            logger.warning("❌ Kein Beispiel-Input verfügbar. TorchScript-Export übersprungen.")
            return
 
        # TorchScript Export
        checkpoint_name = Path(best_checkpoint_path).stem
        ts_path = self.output_dir / f"{checkpoint_name}.pt"
        logger.info(f"🔹 Exportiere nach TorchScript: {ts_path}")
 
        try:
            TorchScriptExportCallback.build_torchscript(pl_module, ts_path, example_input, self.optimize)
        except Exception as e:
            logger.error(f"❌ Fehler beim TorchScript-Export: {e}")
 
    @staticmethod
    def build_torchscript(module, model_path: Path, example_input: torch.Tensor, optimize: bool = True):
        """
        Erstellt ein TorchScript-Modell aus einem PyTorch-Modell.

        :param module: PyTorch-Modul.
        :param model_path: Pfad zum gespeicherten TorchScript-Modell.
        :param example_input: Beispiel-Input für das Modell.
        :param optimize: Falls True, wird `optimize_for_inference` angewendet.
        :return: Gespeichertes TorchScript-Modell.
        """

        try:
            logger.info(f"🔹 Konvertiere Modell nach TorchScript: {model_path}")
            traced_model = torch.jit.trace(module, example_inputs=example_input)
 
            if optimize:
                logger.info("🔹 Wende `optimize_for_inference()` an...")
                traced_model = torch.jit.optimize_for_inference(traced_model)
 
            traced_model.save(model_path)
            logger.info(f"✅ TorchScript-Modell gespeichert: {model_path}")
            return traced_model
 
        except Exception as e:
            logger.error(f"❌ Fehler beim TorchScript-Export: {e}")
            raise e
