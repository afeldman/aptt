from torch import nn

from aptt.model.feature.efficientnet import EfficientNetBackbone


class EfficientNet(EfficientNetBackbone):
    """EfficientNet model for image classification.

    This class implements the EfficientNet architecture with configurable parameters.

    Attributes:
        image_size (tuple): The input image size.
        stem (nn.Sequential): The stem layers.
        blocks (nn.Sequential): The MBConv blocks.
        head (nn.Sequential): The head layers.
        avgpool (nn.AdaptiveAvgPool2d): The adaptive average pooling layer.
        dropout (nn.Dropout): The dropout layer.
        fc (nn.Linear): The fully connected layer.
    """

    def __init__(
        self,
        resolution_coefficient: float,
        width_coefficient: float,
        depth_coefficient: float,
        input_channels: int = 3,
        dropout_rate: float = 0.2,
        num_classes: int = 1000,
        version: str = "b0",
    ):
        """Initialize the EfficientNet model.

        Args:
            resolution_coefficient (float): Coefficient for scaling the input resolution.
            width_coefficient (float): Coefficient for scaling the width (number of channels).
            depth_coefficient (float): Coefficient for scaling the depth (number of layers).
            input_channels (int): Number of input channels. Default is 3.
            dropout_rate (float): Dropout rate. Default is 0.2.
            num_classes (int): Number of output classes. Default is 1000.
            version (str): Version of EfficientNet. Default is "b0".
        """
        super().__init__(
            resolution_coefficient=resolution_coefficient,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            input_channels=input_channels,
            dropout_rate=dropout_rate,
            version=version,
        )

        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        """Forward pass through the EfficientNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = super().forward(x, return_pooled=False, return_stages=False)
        x = self.fc(x)
        return x
