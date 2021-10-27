#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Downsample Stem. """

import torch
import torch.nn as nn

from models.base.base_blocks import Base3DStem
from models.base.base_blocks import STEM_REGISTRY

@STEM_REGISTRY.register()
class DownSampleStem(Base3DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(DownSampleStem, self).__init__(cfg)
        self.maxpool = nn.MaxPool3d(
            kernel_size = (1, 3, 3),
            stride      = (1, 2, 2),
            padding     = (0, 1, 1)
        )
    
    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.maxpool(x)
        return x

