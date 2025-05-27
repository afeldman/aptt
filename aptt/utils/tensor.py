import torch
import numpy as np


def empty_like(x):
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (torch.empty_like(x, dtype=torch.float32) if isinstance(
        x, torch.Tensor) else np.empty_like(x, dtype=np.float32))
