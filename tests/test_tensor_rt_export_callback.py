import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from unittest import mock
from aptt.callbacks.tensor_rt import TensorRTExportCallback


class DummyDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        x = torch.rand(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        return DataLoader(TensorDataset(x, y), batch_size=2)


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


@pytest.fixture
def dummy_trainer(tmp_path):
    model = DummyModel()
    datamodule = DummyDataModule()

    # Fake checkpoint
    checkpoint_path = tmp_path / "best.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)

    callback = TensorRTExportCallback(output_dir=tmp_path)
    callback.best_model_path = str(checkpoint_path)

    trainer = mock.MagicMock()
    trainer.datamodule = datamodule
    trainer.model = model

    return trainer, model, callback, checkpoint_path


@mock.patch("aptt.callbacks.tensor_rt.TorchScriptExportCallback.build_torchscript")
@mock.patch("aptt.callbacks.tensor_rt.torch_tensorrt.ts.compile")
@mock.patch("aptt.callbacks.tensor_rt.torch.jit.save")
@mock.patch("aptt.callbacks.tensor_rt.torch.jit.load")
def test_tensor_rt_export(
    mock_jit_load,
    mock_jit_save,
    mock_tensorrt_compile,
    mock_build_torchscript,
    dummy_trainer,
):
    trainer, model, callback, ckpt_path = dummy_trainer

    mock_jit_load.return_value = model  # simulate loading TS model
    mock_tensorrt_compile.return_value = model  # simulate TRT conversion

    callback.on_validation_end(trainer, model)

    mock_build_torchscript.assert_called_once()
    mock_tensorrt_compile.assert_called_once()
    mock_jit_save.assert_called_once()
