import torch
from torch import nn

from aptt.model.feature.mobile import MobileNetV1Backbone, MobileNetV2Backbone, MobileNetV3Backbone


class MobileNetV1(MobileNetV1Backbone):
    """MobileNetV1 architecture.

    Args:
        num_classes (int, optional): output neuron in last layer. Defaults to 1000.
        width_multiplier (float, optional): width multiplier for MobileNetV1. Defaults to 1.0.

    Examples:
        >>> model = MobileNetV1()
        >>> output = model(torch.rand(1, 3, 224, 224))

    Raises:
        TypeError: If input is not a tensor
    """

    def __init__(self, num_classes=1000, width_multiplier=1.0, stage_indices=(3, 6, 11)):
        super().__init__(width_multiplier=width_multiplier, stage_indices=stage_indices)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(1024 * width_multiplier), num_classes)

    def forward(self, x: torch.Tensor):
        x = super().forward(x, return_stages=False, return_pooled=False)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class MobileNetV2(MobileNetV2Backbone):
    """MobileNetV2 architecture.

    Args:
        n_classes (int, optional): output neuron in last layer. Defaults to 1000.
        input_channel (int, optional): input channels in first conv layer. Defaults to 3.
        dropout (float, optional): dropout in last layer. Defaults to 0.2.

    Examples:
        >>> model = MobileNetV2()
        >>> output = model(torch.rand(1, 3, 224, 224))
    """

    def __init__(self, n_classes=1000, dropout=0.2, stage_indices=(3, 6, 13)):
        super().__init__(stage_indices=stage_indices, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, n_classes)

    def forward(self, x):
        x = super().forward(x, return_stages=False, return_pooled=False)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MobileNetV3(MobileNetV3Backbone):
    """MobileNetV3 architecture.

    Args:
        n_classes (int, optional): output neuron in last layer. Defaults to 1000.
        input_channel (int, optional): input channels in first conv layer. Defaults to 3.
        config (str, optional): configuration of MobileNetV3, either `large` or `small`. Defaults to `large`.
        dropout (float, optional): dropout in last layer. Defaults to 0.8.

    Raises:
        TypeError: If input is not a tensor

    Examples:
        >>> model = MobileNetV3()
        >>> output = model(torch.rand(1, 3, 224, 224))
    """

    def __init__(
        self,
        n_classes=1000,
        config="large",
        dropout=0.8,
        stage_indices=(3, 6, 11),
    ):
        super().__init__(config=config, stage_indices=stage_indices, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1280 if config == "large" else 1024, n_classes)

    def forward(self, x):
        x = super().forward(x, return_stages=False, return_pooled=False)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
