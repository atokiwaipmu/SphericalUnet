
from traceback import print_tb
import torch
from torch import nn
import pytorch_lightning as pl
from functools import partial

from scripts.layers.cheby_shev import SphericalChebConv

from scripts.layers.normalization import Norms
from scripts.layers.activation import Acts
from scripts.blocks.Resnetblock import ResnetBlock
from scripts.utils.partial_laplacians import get_partial_laplacians
from scripts.utils.healpix_pool_unpool import Healpix
    
class Unet(pl.LightningModule):
    """
    Full Unet architecture composed of an encoder (downsampler), a bottleneck, and a decoder (upsampler).
    The architecture is inspired by the one used in the (https://arxiv.org/abs/2311.05217).
    """
    def __init__(self, params):
        super().__init__()

        # params for basic architecture
        self.dim_in = params["architecture"]["dim_in"]    
        self.dim_out = params["architecture"]["dim_out"]
        self.dim = params["architecture"]["inner_dim"]
        self.dim_mults = [self.dim * factor for factor in params["architecture"]["mults"]]
        self.num_resblocks = params["architecture"]["num_blocks"]

        # params for further customization
        self.norm_type = params["architecture"]["norm_type"]
        self.act_type = params["architecture"]["act_type"]
        self.skip_factor = params["architecture"]["skip_factor"]
        self.use_conv = params["architecture"]["use_conv"]

        # params for Healpix
        self.kernel_size = params["architecture"]["kernel_size"]
        self.nside = params["data"]["nside"]
        self.order = params["data"]["order"]
        self.pooling = Healpix().pooling
        self.unpooling = Healpix().unpooling
        self.depth = len(self.dim_mults)
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        self.init_conv = SphericalChebConv(self.dim_in, self.dim, self.laps[-1], kernel_size=20)

        block_partial = partial(ResnetBlock, 
                        kernel_size=self.kernel_size,
                        norm_type=self.norm_type,
                        act_type=self.act_type, 
                        use_conv=self.use_conv)

        self.down_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap_id, lap_down, num_resblock in zip(self.dim_mults[:-1], self.dim_mults[1:], reversed(self.laps[1:]), reversed(self.laps[:-1]), reversed(self.num_resblocks)):
            tmp = nn.ModuleList([])
            for jj in range(num_resblock):
                tmp.append(block_partial(dim_in, dim_in, lap_id))
            tmp.append(block_partial(dim_in, dim_out, lap_down, down = True))
            self.down_blocks.append(tmp)

        self.mid_block1 = block_partial(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])
        self.mid_block2 = block_partial(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])

        self.up_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap_id, lap_up, num_resblock in zip(reversed(self.dim_mults[1:]), reversed(self.dim_mults[:-1]), self.laps[:-1], self.laps[1:], self.num_resblocks):
            tmp = nn.ModuleList([])
            tmp.append(block_partial(2*dim_in, dim_in, lap_id))
            tmp.append(block_partial(dim_in, dim_out, lap_up, up = True))
            self.up_blocks.append(tmp)

        self.out_block = block_partial(2 * self.dim_mults[0], self.dim, self.laps[-1])
        self.out_norm = Norms(self.dim, self.norm_type)
        self.out_act = Acts(self.act_type)
        self.final_conv = SphericalChebConv(self.dim, self.dim_out, self.laps[-1], kernel_size=20)

    def forward(self, x):
        skip_connections = []
        x = self.init_conv(x)
        skip_connections.append(x)

        # downsample
        for downs in self.down_blocks:
            for block in downs:
                x = block(x)
            skip_connections.append(x)

        # bottleneck
        x = self.mid_block1(x)
        x = self.mid_block2(x)

        # upsample
        for ups in self.up_blocks:
            tmp_connection = skip_connections.pop() * self.skip_factor
            x = torch.cat([x, tmp_connection], dim=2)
            for block in ups:
                x = block(x)

        tmp_connection = skip_connections.pop() * self.skip_factor
        x = torch.cat([x, tmp_connection], dim=2)
        x = self.out_block(x)
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.final_conv(x)
        return x