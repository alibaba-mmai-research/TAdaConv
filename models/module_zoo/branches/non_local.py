#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" NonLocal block. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.base_blocks import BaseBranch, BRANCH_REGISTRY

@BRANCH_REGISTRY.register()
class NonLocal(BaseBranch):
    """
    Non-local block.
    
    See Xiaolong Wang et al.
    Non-local Neural Networks.
    """

    def __init__(self, cfg, block_idx):
        super(NonLocal, self).__init__(cfg, block_idx)

        self.dim_middle = self.dim_in // 2

        self.qconv = nn.Conv3d(
            self.dim_in,
            self.dim_middle,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.kconv = nn.Conv3d(
            self.dim_in,
            self.dim_middle,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.vconv = nn.Conv3d(
            self.dim_in,
            self.dim_middle,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.out_conv = nn.Conv3d(
            self.dim_middle,
            self.num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.out_bn = nn.BatchNorm3d(self.num_filters, eps=1e-5, momentum=self.bn_mmt)

    def forward(self, x):
        n,c,t,h,w = x.shape

        query = self.qconv(x).view(n, self.dim_middle, -1)
        key = self.kconv(x).view(n, self.dim_middle, -1)
        value = self.vconv(x).view(n, self.dim_middle, -1)

        attn = torch.einsum("nct,ncp->ntp", (query, key))
        attn = attn * (self.dim_middle ** -0.5)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum("ntg,ncg->nct", (attn, value))
        out = out.view(n, self.dim_middle, t, h, w)
        out = self.out_conv(out)
        out = self.out_bn(out)
        return x + out


