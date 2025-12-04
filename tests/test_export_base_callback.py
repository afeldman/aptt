import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from aptt.callbacks.base import ExportBaseCallback


class DummyDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        x = torch.rand(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        return DataLoader(TensorDataset(x, y), batch_size=4)


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


def test_get_example_input(tmp_path):
    model = DummyModel()
    datamodule = DummyDataModule()
    callback = ExportBaseCallback(output_dir=tmp_path)

    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_model_summary=False,
        callbacks=[callback],
    )

    trainer.datamodule = datamodule

    example_input = callback.get_example_input(trainer)
    assert example_input is not None
    assert isinstance(example_input, torch.Tensor)
    assert example_input.shape[1:] == (3, 32, 32)
    assert example_input.device == model.device
