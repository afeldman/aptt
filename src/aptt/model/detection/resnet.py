import torch
from torch import nn

from aptt.model.feature.resnet import ResNetBackbone


class ResNet(ResNetBackbone):
    def __init__(self, resnet_variant, in_channels, num_classes, stage_indices=(3, 6, 11)):
        """Creates the ResNet architecture based on the provided variant. 18/34/50/101 etc.
        Based on the input parameters, define the channels list, repeatition list along with expansion factor(4) and stride(3/1)
        using _make_blocks method, create a sequence of multiple Bottlenecks
        Average Pool at the end before the FC layer.

        Args:
            resnet_variant (list) : eg. [[64,128,256,512],[3,4,6,3],4,True]
            in_channels (int) : image channels (3)
            num_classes (int) : output #classes

        Attributes:
            Layer consisting of conv->batchnorm->relu

        """
        super().__init__(resnet_variant=resnet_variant, in_channels=in_channels, stage_indices=stage_indices)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.channels_list[3] * self.expansion, num_classes)

    def forward(self, x):
        x = super().forward(x, return_stages=False, return_pooled=False)

        x = self.average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x
