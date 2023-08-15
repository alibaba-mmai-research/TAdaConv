#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" TAdaFormer block. """

import torch
import torch.nn as nn

from collections import OrderedDict

from tadaconv.models.module_zoo.ops.misc import QuickGELU
from tadaconv.models.utils.init_helper import trunc_normal_
from tadaconv.models.base.base_blocks import BRANCH_REGISTRY, DropPath
from tadaconv.models.module_zoo.ops.tadaconv_v2 import TAdaConv2dV2


class TAdaBlockTempEnhanced(nn.Module):
    r""" TAdaFormer Block. 

    Args: 
        cfg (Config): the global config object.
    """
    def __init__(self, cfg):
        super().__init__()
        backbone_cfg = cfg.VIDEO.BACKBONE

        num_frames = cfg.DATA.NUM_INPUT_FRAMES // backbone_cfg.TUBLET_STRIDE
        
        d_model = backbone_cfg.NUM_FEATURES
        d_middle = int(d_model//backbone_cfg.REDUCTION)

        self.tada_pre_proj = nn.Conv3d(
            in_channels=d_model,
            out_channels=d_middle,
            kernel_size=(3,1,1),
            padding=(1,0,0),
        )
        self.tadaconv = TAdaConv2dV2(
            d_middle, d_middle, kernel_size=(1,3,3), padding=(0,1,1), groups=d_middle,
            cal_dim="cout", num_frames=num_frames, rf_r=backbone_cfg.BRANCH.ROUTE_FUNC_R, rf_k=backbone_cfg.BRANCH.ROUTE_FUNC_K, bias=True
        )
        self.gelu = QuickGELU()
        self.tada_proj = nn.Conv3d(
            in_channels=d_middle,
            out_channels=d_model,
            kernel_size=1,
            padding=0
            )

    def forward(self, x):
        h = w = int(x.shape[0]**0.5)
        # L, N, C -> H, W, B, T, C
        x = x.reshape(h, w, -1, self.tadaconv.num_frames, x.shape[-1]).permute(2,4,3,0,1)
        x = self.tada_pre_proj(x)
        x = self.tadaconv(x,reshape_required=False)
        x = self.gelu(x)
        x = self.tada_proj(x)
        x = x.permute(3,4,0,2,1).reshape(h*w,-1,x.shape[1])
        return x

class TAdaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone_cfg = cfg.VIDEO.BACKBONE

        num_frames = cfg.DATA.NUM_INPUT_FRAMES // backbone_cfg.TUBLET_STRIDE

        d_model = backbone_cfg.NUM_FEATURES
        d_middle = int(d_model//backbone_cfg.REDUCTION)

        self.tada_pre_proj = nn.Linear(d_model, d_middle)
        self.tadaconv = TAdaConv2dV2(
            d_middle, d_middle, kernel_size=(1,3,3), padding=(0,1,1), groups=d_middle,
            cal_dim="cout", num_frames=num_frames, rf_r=backbone_cfg.BRANCH.ROUTE_FUNC_R, rf_k=backbone_cfg.BRANCH.ROUTE_FUNC_K, bias=True
        )
        self.gelu = QuickGELU()
        self.tada_proj = nn.Linear(d_middle, d_model)

    def forward(self, x):
        x = self.tada_pre_proj(x)
        x = self.tadaconv(x)
        x = self.gelu(x)
        x = self.tada_proj(x)
        return x

@BRANCH_REGISTRY.register()
class TAdaFormerBlock(nn.Module):
    """
    Transformer layer for (L, N, D) input.
    Modified from CLIP.
    """
    def __init__(self, cfg, drop_path_rate: float = 0.0):
        super().__init__()

        # no scaling anymore from V5_V2

        backbone_cfg = cfg.VIDEO.BACKBONE

        num_frames = cfg.DATA.NUM_INPUT_FRAMES // backbone_cfg.TUBLET_STRIDE

        d_model = backbone_cfg.NUM_FEATURES
        n_head = backbone_cfg.NUM_HEADS
        attn_dropout = backbone_cfg.ATTN_DROPOUT

        temporal_enhance = backbone_cfg.TEMP_ENHANCE
        tadablock = TAdaBlockTempEnhanced if temporal_enhance else TAdaBlock

        self.double_tada = backbone_cfg.DOUBLE_TADA

        self.tada = tadablock(cfg)
        if self.double_tada:
            self.tada2 = tadablock(cfg)

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

        self.drop_path = DropPath(drop_path_rate,mode="L(BT)C",num_frames=num_frames) if drop_path_rate > 0. else nn.Identity()

        self.apply(self._init_weights)
        # zero init the last projection layer of tadablocks
        with torch.no_grad():
            self.tada.tada_proj.weight.data.zero_()
            self.tada.tada_proj.bias.data.zero_()
            if self.double_tada:
                self.tada2.tada_proj.weight.data.zero_()
                self.tada2.tada_proj.bias.data.zero_()
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            if hasattr(m, "skip_init") and m.skip_init:
                return
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(torch.cat((torch.zeros_like(x[0:1], device=x.device),self.tada(x[1:])), dim=0))
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        if self.double_tada:
            x = x + self.drop_path(torch.cat((torch.zeros_like(x[0:1], device=x.device),self.tada2(x[1:])), dim=0))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x