#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" TAdaConvNeXtV2 block. """

import torch
import torch.nn as nn

from collections import OrderedDict

from tadaconv.models.module_zoo.ops.misc import QuickGELU
from tadaconv.models.utils.init_helper import trunc_normal_
from tadaconv.models.base.base_blocks import BRANCH_REGISTRY, DropPath
from tadaconv.models.module_zoo.ops.tadaconv_v2 import TAdaConv2dV2, RouteFuncwTransformer
from tadaconv.models.module_zoo.ops.misc import LayerNorm

@BRANCH_REGISTRY.register()
class TAdaConvNeXtV2Block(nn.Module):
    r""" TAdaConvNeXtV2 Block. 
    Args:
        cfg (Config): the global config object.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, cfg, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = TAdaConv2dV2(
            dim, dim, kernel_size=(1,7,7), padding=(0,3,3), groups=dim,
            cal_dim="cout", 
            internal_rf_func=False,
            internal_temp_aggr=False
        )
        self.dwconv_rf = RouteFuncwTransformer(
            c_in=dim,
            ratio=cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_R,
            kernels=cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_K,
            with_bias_cal=self.dwconv.bias is not None,
            zero_init_cal=True,
            head_dim=cfg.VIDEO.BACKBONE.BRANCH.HEAD_DIM if hasattr(cfg.VIDEO.BACKBONE.BRANCH, "HEAD_DIM") else 48
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.avgpool = nn.AvgPool3d(kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0))
        self.norm_avgpool = LayerNorm(dim, eps=1e-6)
        self.norm_avgpool.weight.data.zero_()
        self.norm_avgpool.bias.data.zero_()
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = QuickGELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.dwconv(x, reshape_required=False, alpha=self.dwconv_rf(x))

        # temporal aggregation
        norm_avgpool_x = self.avgpool(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, T, H, W) -> (N, T, H, W, C)
        norm_avgpool_x = norm_avgpool_x.permute(0, 2, 3, 4, 1) # (N, C, T, H, W) -> (N, T, H, W, C)
        x = self.norm(x) + self.norm_avgpool(norm_avgpool_x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, T, H, W, C) -> (N, C, T, H, W)

        x = input + self.drop_path(x)
        return x