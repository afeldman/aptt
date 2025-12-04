"""TensorRT export callback for PyTorch Lightning.

This module contains a callback that exports the best model to TensorRT
once it has been saved. Only available on Linux and Windows.

Example:
    ```python
    from pytorch_lightning import Trainer
    from aptt.callbacks.tensor_rt import TensorRTExportCallback

    trainer = Trainer(callbacks=[TensorRTExportCallback()])
    trainer.fit(model)
    ```

Note:
    TensorRT is only supported on Linux and Windows platforms.
    This callback will raise an ImportError on macOS.
"""

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger

from aptt.callbacks.base import ExportBaseCallback
from aptt.callbacks.torchscript import TorchScriptExportCallback

# TensorRT is only available on Linux and Windows
if sys.platform in ("linux", "win32"):
    try:
        import torch_tensorrt  # type: ignore
        TENSORRT_AVAILABLE = True
    except ImportError:
        TENSORRT_AVAILABLE = False
        logger.warning("torch_tensorrt not available. Install with: pip install torch-tensorrt")
else:
    TENSORRT_AVAILABLE = False
    torch_tensorrt = None  # type: ignore
    logger.info(f"TensorRT is not supported on {sys.platform}. Only available on Linux/Windows.")


class TensorRTExportCallback(ExportBaseCallback):
    """Exports the best model to TensorRT (.trt) using TorchScript.
    
    Note:
        Only available on Linux and Windows. On macOS, this callback will
        raise an error during initialization.
    """

    def __init__(self, output_dir="models", precision="fp16", workspace_size=1 << 20, **kwargs):
        """Initialize the TensorRT export callback.

        Args:
            output_dir (str): Directory where exported models will be saved.
            precision (str): TensorRT precision mode ('fp16', 'fp32', 'int8').
            workspace_size (int): Workspace size in bytes for TensorRT.
            **kwargs: Additional arguments passed to the base ModelCheckpoint.
            
        Raises:
            RuntimeError: If TensorRT is not available on this platform.
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                f"TensorRT is not available on {sys.platform}. "
                "Only supported on Linux and Windows. "
                "Install with: pip install 'aptt[tensorrt]' (Linux/Windows only)"
            )
        
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.precision = precision
        self.workspace_size = workspace_size

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Export the best model to TensorRT (.trt) via TorchScript after validation ends.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): Model being trained.

        Example:
            ```python
            from pytorch_lightning import Trainer
            from aptt.callbacks.tensor_rt import TensorRTExportCallback

            trainer = Trainer(callbacks=[TensorRTExportCallback()])
            trainer.fit(model)
            ```
        """
        super().on_validation_end(trainer, pl_module)

        best_checkpoint_path = self.best_model_path
        if not best_checkpoint_path or not Path(best_checkpoint_path).exists():
            logger.warning("❌ No valid checkpoint found. Skipping export.")
            return

        logger.info(f"🔹 Loading best model: {best_checkpoint_path}")

        pl_module.load_state_dict(torch.load(best_checkpoint_path, map_location="cpu")["state_dict"])
        pl_module.eval()

        example_input = self.get_example_input(trainer)
        if example_input is None:
            logger.warning("❌ No example input available. Skipping TensorRT export.")
            return

        checkpoint_name = Path(best_checkpoint_path).stem
        ts_path = self.output_dir / f"{checkpoint_name}.pt"

        # Export TorchScript model
        if not ts_path.exists():
            logger.info(f"🔹 Exporting TorchScript model: {ts_path}")
            try:
                TorchScriptExportCallback.build_torchscript(
                    pl_module,
                    ts_path,
                    example_input,
                    optimize=True,
                )
            except Exception as e:
                logger.error(f"❌ TorchScript export failed: {e}")
                return
        else:
            logger.info(f"✅ TorchScript model already exists: {ts_path}")

        # Load and verify TorchScript model
        try:
            torchscript_model = torch.jit.load(ts_path)
            logger.info(f"✅ TorchScript model loaded: {ts_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load TorchScript model: {e}")
            return

        # Convert to TensorRT
        trt_path = self.output_dir / f"{checkpoint_name}.trt"
        logger.info(f"🔹 Converting TorchScript to TensorRT: {trt_path}")

        try:
            precision_map = {"fp32": torch.float32, "fp16": torch.float16, "int8": torch.int8}
            precision_type = precision_map.get(self.precision, torch.float16)

            trt_model = torch_tensorrt.ts.compile(
                torchscript_model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=example_input.shape,
                        opt_shape=example_input.shape,
                        max_shape=example_input.shape,
                    )
                ],
                enabled_precisions={precision_type},
                workspace_size=self.workspace_size,
            )

            torch.jit.save(trt_model, trt_path)
            logger.info(f"✅ TensorRT model saved: {trt_path}")
        except Exception as e:
            logger.error(f"❌ TensorRT export failed: {e}")
