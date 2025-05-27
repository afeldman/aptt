import torch
from torch import nn

# Bounding Box Head (Regressor)
class BBoxHead(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()
        # 5 = (Cx, Cy, W, H, θ) für eine ankerlose Box pro Feature-Map-Zelle
        self.conv = nn.Conv2d(in_channels, 5, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B, 5, H, W)
