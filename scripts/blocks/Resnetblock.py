#Description: This file contains the implementation of the ResnetBlock class, which is used to define the ResNet architecture.
from torch import nn
from scripts.layers.cheby_shev import SphericalChebConv
from scripts.layers.normalization import Norms
from scripts.layers.activation import Acts
from scripts.utils.healpix_pool_unpool import Healpix
from scripts.utils.utils import exists

'''
The code defines various neural network blocks, specifically for ResNet architecture.
These blocks include standard convolutional layers, normalization, and activation functions.
The implementation is designed to be modular, allowing easy integration into larger neural network models.
'''

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, 
                laplacian, kernel_size=20, dropout=0.0,
                norm_type="batch", act_type="silu"):
        super().__init__()
        self.block = nn.Sequential(
            Norms(in_channels, norm_type),
            Acts(act_type) if out_channels > 1 else nn.Identity(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        )

    def forward(self, x):
        return self.block(x)
    
class ResnetBlock(nn.Module):
    """
    Up/Downsampling Residual block implemented from https://arxiv.org/abs/2311.05217.
    Originally https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, in_channels, out_channels, laplacian, 
                kernel_size=20, dropout=0.0, 
                norm_type="batch", act_type="silu", 
                use_conv=False, up=False, down=False):
        super().__init__()
        
        self.in_layers = Block(in_channels, out_channels, laplacian, kernel_size, dropout, norm_type, act_type)

        self.updown = up or down
        if up:
            self.pooling = Healpix().unpooling
        elif down:
            self.pooling = Healpix().pooling
        else:
            self.pooling = nn.Identity()

        self.out_layers = Block(out_channels, out_channels, laplacian, kernel_size, dropout, norm_type, act_type)

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        else:
            self.skip_connection = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers.block[:-1], self.in_layers.block[-1]
            h = in_rest(x)
            h = self.pooling(h)
            h = in_conv(h)
            x = self.pooling(x)
        else:
            h = self.in_layers(x)
            
        h = self.out_layers(h)

        return self.skip_connection(x) + h