import numpy as np
import torch
import torch.nn as nn


class CCM(nn.Module):
    """Channel Coupling Module"""

    def __init__(self, pool_size: int, img_size: int, hidden_size: int = ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(img_size**2, )
        )



class CLCM(nn.Module):
    """Convolutional Layer Coupling Module."""

    def __init__(self, conv_channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(1, conv_channels, 3)

