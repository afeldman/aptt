from torch import nn


class DetectionHead(nn.Module):
    """Detection-Head für Bounding Boxes, Klassen und Objektness-Scores."""

    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(in_channels, (5 + num_classes) * num_anchors, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
