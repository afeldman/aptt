from copy import deepcopy
from typing import Optional

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.model_summary import ModelSummary

from apu.ml.callbacks.tensor_rt import TensorRTExportCallback
from apu.ml.callbacks.torchscript import TorchScriptExportCallback
from apu.ml.config.config_manager import ConfigManager


class BaseTrainer(pl.Trainer):
    """
    Basisklasse für Trainer in PyTorch Lightning.

    Diese Klasse erweitert die Standardfunktionalität von PyTorch Lightning um folgende Features:
    - Automatische Batch-Größen-Optimierung
    - Integration von MLflow für Experiment-Logging
    - Export des Modells nach ONNX, TensorRT und TorchScript (optional)
    - Anzeige eines Rich Progress Bar während des Trainings

    Args:
        log_dir (str): Verzeichnis für Log-Dateien.
        use_mlflow (bool): Ob MLflow für Experiment-Logging verwendet werden soll.
        mlflow_experiment (str): Name des MLflow-Experiments.
        auto_batch_size (bool): Automatische Batch-Größe aktivieren.
        model_output_dir (str): Verzeichnis für gespeicherte Modell-Exporte.
        export_formats (List[str], optional): Liste der aktivierten Export-Formate, z. B. ["onnx", "tensor_rt"].
        onnx_opset_version (int, optional): OpSet-Version für ONNX-Export (Standard: 11).
        **kwargs: Weitere Argumente für den Trainer.

    Attributes:
        auto_batch_size (bool): Automatische Batch-Größe aktiv.

    Example:
        trainer = BaseTrainer(log_dir="logs", use_mlflow=True, mlflow_experiment="my_experiment")
        trainer.fit(model, datamodule=datamodule)
    """

    def __init__(
        self,
        log_dir: str = "logs",
        auto_batch_size: bool = True,  # Automatische Batch-Größe aktivieren
        use_mlflow: bool = False,
        mlflow_experiment: Optional[str] = None,
        model_output_dir: str = "models",
        export_formats: Optional[list[str]] = None,
        onnx_opset_version: int = 11,  # Standardmäßig OpSet 11 verwenden
        early_stopping: Optional[EarlyStopping] = None,
        model_checkpoint: Optional[ModelCheckpoint] = None,
        **kwargs,
    ):
        # Callbacks initialisieren
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(RichProgressBar())

        activated_callbacks = []

        # Falls keine Export-Formate angegeben sind, werden alle aktiviert
        if export_formats is None:
            export_formats = ["onnx"]

        if "onnx" in export_formats:
            callbacks.append(
                ONNXExportCallback(output_dir=model_output_dir, simplify=True, opversion=onnx_opset_version)
            )
            activated_callbacks.append(f"ONNX (OpSet {onnx_opset_version})")

        if "tensor_rt" in export_formats:
            callbacks.append(TensorRTExportCallback(output_dir=model_output_dir))
            activated_callbacks.append("TensorRT")

        if "torchscript" in export_formats:
            callbacks.append(TorchScriptExportCallback(output_dir=model_output_dir))
            activated_callbacks.append("TorchScript")

        if not early_stopping:
            early_stopping = EarlyStopping(monitor="val_acc", patience=5, mode="max")
            logger.info(f"ℹ️  EarlyStopping aktiviert: Monitor='val_acc', Patience=5")
        else:
            logger.info(f"ℹ️  Custom EarlyStopping verwendet.")

        if not model_checkpoint:
            model_checkpoint = ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                dirpath="models",
                filename="best-checkpoint_{epoch:03d}-{val_acc:.4f}",
            )
            logger.info(f"ℹ️  ModelCheckpoint aktiviert: Monitor='val_acc', Top-1, Pfad='models'")
        else:
            logger.info(f"ℹ️  Custom ModelCheckpoint verwendet.")

        # Logge aktivierte Callbacks
        if activated_callbacks:
            logger.info(f"🔹 Aktivierte Export-Callbacks: {', '.join(activated_callbacks)}")
        else:
            logger.info("⚠️ Keine Export-Callbacks aktiviert.")

        # Standard-Logger: TensorBoard
        loggers: list[Logger] = [TensorBoardLogger(save_dir=log_dir, name="tensorboard")]

        # Optional: MLflow
        if use_mlflow:
            if not mlflow_experiment:
                mlflow_experiment = "lightning_mlflow"
            mlflow_logger = MLFlowLogger(experiment_name=mlflow_experiment)
            loggers.append(mlflow_logger)

        super().__init__(logger=loggers, callbacks=callbacks, accelerator="gpu", devices="auto", **kwargs)

        # Falls gewünscht, automatische Batch-Größe optimieren
        self.auto_batch_size = auto_batch_size

    def tune_batch_size(
        self, model: pl.LightningModule, datamodule: Optional[pl.LightningDataModule] = None
    ) -> Optional[int]:
        """
        Optimiert die Batch-Größe des Modells.

        Args:
            model: PyTorch-Modell.
            datamodule: PyTorch Lightning DataModule.

        Returns:
            int: Optimierte Batch-Größe oder None.

        Example:
            trainer = BaseTrainer(log_dir="logs", use_mlflow=True)
            new_batch_size = trainer.tune_batch_size(model, datamodule=datamodule)
        """
        if self.auto_batch_size:
            tuner = Tuner(self)
            new_batch_size = tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
            logger.info(f"🔹 Optimierte Batch-Größe: {new_batch_size}")
            return new_batch_size
        return None

    def fit(self, model, *args, **kwargs):
        """
        Trainiert das Modell.

        Args:
            model: PyTorch-Modell.
            *args: Weitere Argumente für den Trainer.
            **kwargs: Weitere Argumente für den Trainer.

        Example:
            trainer = BaseTrainer(log_dir="logs", use_mlflow=True)
            trainer.fit(model, datamodule=datamodule)
        """
        # Model Summary beim Start ausgeben
        model_summary = ModelSummary(model, max_depth=3)
        logger.info("\n" + "=" * 50)
        logger.info("MODULE SUMMARY")
        logger.info("=" * 50)
        logger.info(str(model_summary))
        logger.info("=" * 50 + "\n")

        super().fit(model, *args, **kwargs)
        logger.info("🔹 Training abgeschlossen.\n" + "=" * 50)
