"""
TensorRT Export Callback für PyTorch Lightning.

Dieses Modul enthält einen Callback, der das beste Modell nach TensorRT exportiert, sobald es gespeichert wurde.

Beispiel:
```python
from pytorch_lightning import Trainer
from apu.ml.callbacks.tensor_rt import TensorRTExportCallback

trainer = Trainer(callbacks=[TensorRTExportCallback()])
trainer.fit(model)
```
"""

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch_tensorrt  # type: ignore
from loguru import logger

from apu.ml.callbacks.base import ExportBaseCallback
from apu.ml.callbacks.torchscript import TorchScriptExportCallback


class TensorRTExportCallback(ExportBaseCallback):
    """
    Exportiert das Modell nach TensorRT (.trt) über TorchScript.
    """

    def __init__(self, output_dir="models", precision="fp16", workspace_size=1 << 20, **kwargs):
        """
        :param output_dir: Verzeichnis für exportierte Modelle.
        :param precision: TensorRT Präzision (fp16, fp32, int8).
        :param workspace_size: Speichergröße für TensorRT.
        :param kwargs: Weitere Argumente für ModelCheckpoint.
        """
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.precision = precision
        self.workspace_size = workspace_size

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Exportiert das Modell nach TensorRT (.trt) über TorchScript.

        :param trainer: PyTorch Lightning Trainer.
        :param pl_module: PyTorch Lightning

        :example:
        ```python
        from pytorch_lightning import Trainer
        from apu.ml.checkpoint import TensorRTExportCallback

        trainer = Trainer(callbacks=[TensorRTExportCallback()])
        trainer.fit(model)
        ```
        """
        super().on_validation_end(trainer, pl_module)

        best_checkpoint_path = self.best_model_path
        if not best_checkpoint_path or not Path(best_checkpoint_path).exists():
            logger.warning("❌ Kein gültiger Checkpoint gefunden. Export übersprungen.")
            return

        logger.info(f"🔹 Lade bestes Modell: {best_checkpoint_path}")

        pl_module.load_state_dict(torch.load(best_checkpoint_path, map_location="cpu")["state_dict"])
        pl_module.eval()

        # Beispiel-Input automatisch holen
        example_input = self.get_example_input(trainer)
        if example_input is None:
            logger.warning("❌ Kein Beispiel-Input verfügbar. TensorRT-Export übersprungen.")
            return

        checkpoint_name = Path(best_checkpoint_path).stem
        ts_path = self.output_dir / f"{checkpoint_name}.pt"

        # **TorchScript exportieren**
        if not ts_path.exists():
            logger.info(f"🔹 Exportiere TorchScript-Modell: {ts_path}")
            try:
                TorchScriptExportCallback.build_torchscript(pl_module, ts_path, example_input, optimize=True)
            except Exception as e:
                logger.error(f"❌ Fehler beim TorchScript-Export: {e}")
                return
        else:
            logger.info(f"✅ TorchScript-Modell existiert bereits: {ts_path}")

        # **Prüfe, ob das TorchScript-Modell geladen werden kann**
        try:
            torchscript_model = torch.jit.load(ts_path)
            logger.info(f"✅ TorchScript-Modell erfolgreich geladen: {ts_path}")
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von TorchScript: {e}")
            return

        # **TensorRT Export**
        trt_path = self.output_dir / f"{checkpoint_name}.trt"
        logger.info(f"🔹 Konvertiere TorchScript zu TensorRT: {trt_path}")

        try:
            precision_map = {"fp32": torch.float32, "fp16": torch.float16, "int8": torch.int8}
            precision_type = precision_map.get(self.precision, torch.float16)

            trt_model = torch_tensorrt.ts.compile(
                torchscript_model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=example_input.shape, opt_shape=example_input.shape, max_shape=example_input.shape
                    )
                ],
                enabled_precisions={precision_type},
                workspace_size=self.workspace_size,
            )

            torch.jit.save(trt_model, trt_path)
            logger.info(f"✅ TensorRT-Modell gespeichert: {trt_path}")
        except Exception as e:
            logger.error(f"❌ Fehler beim TensorRT-Export: {e}")
