#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" TAdaConvV2. """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

from .misc import LayerNorm, QuickGELU
from tadaconv.models.utils.init_helper import trunc_normal_


class RouteFuncwTransformer(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio, kernels, with_bias_cal=False, bn_eps=1e-5, bn_mmt=0.1, zero_init_cal=True, head_dim=64):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super().__init__()
        self.c_in = c_in
        self.head_dim = head_dim
        self.with_bias_cal = with_bias_cal
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernels[0],1,1],
            padding=[kernels[0]//2,0,0],
        )

        self.norm = LayerNorm(int(c_in//ratio), eps=1e-6, data_format="channels_first")
        self.norm_transformer = LayerNorm(int(c_in//ratio), eps=1e-6, data_format="channels_first")
        self.gelu = QuickGELU()

        self.scale = int(c_in//ratio) ** -0.5
        self.qkv_proj = nn.Conv3d(
            in_channels = int(c_in//ratio),
            out_channels = int(c_in//ratio)*3,
            kernel_size=1,
            padding=0,
        )

        self.attn_out = nn.Conv3d(
            in_channels = int(c_in//ratio),
            out_channels = int(c_in//ratio),
            kernel_size=1,
            padding=0,
        )

        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernels[1],1,1],
            padding=[kernels[1]//2,0,0],
            bias=False
        )
        self.zero_init_cal = zero_init_cal
        if zero_init_cal:
            self.b.skip_init=True
            self.b.weight.data.zero_() # to make sure the initial values 
                                    # for the output is 1.
        if with_bias_cal:
            self.b_bias = nn.Conv3d(
                in_channels=int(c_in//ratio),
                out_channels=c_in,
                kernel_size=[kernels[1],1,1],
                padding=[kernels[1]//2,0,0],
                bias=False
            )
            if zero_init_cal:
                self.b_bias.skip_init=True
                self.b_bias.weight.data.zero_() # to make sure the initial values 
                                        # for the output is 1.
            
    def forward(self, x):
        x = self.avgpool(x)
        x = self.a(x)
        x = self.norm(x)
        x = self.gelu(x)

        x = x + self.forward_attention(self.norm_transformer(x))

        if self.with_bias_cal:
            if self.zero_init_cal:
                return [self.b(x) + 1, self.b_bias(x) + 1]
            else:
                return [self.b(x), self.b_bias(x)]
        else:
            if self.zero_init_cal:
                return self.b(x) + 1
            else:
                return self.b(x)
    
    def forward_attention(self, x):
        b, c, t, _, _ = x.shape
        qkv = self.qkv_proj(x)[:,:,:,0,0].view(b,3,self.head_dim,c//self.head_dim,t).permute(1,0,3,4,2)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(-2,-1).reshape(b, c, t)[:,:,:,None,None]

        x = self.attn_out(x)
        
        return x

class TAdaConv2dV2(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d 
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin", num_frames=None, rf_r=4, rf_k=[3,3], head_dim=64, 
                 internal_rf_func=True, internal_temp_aggr=True):
        super().__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
             padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
            cal_dim (str): calibrated dimension in TAdaConv2d. 
                Supported input "cin", "cout".
            head_dim (int): head dimension for MHA in the rourting function.
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        self.num_frames = num_frames

        if internal_rf_func:
            self.rf_func = RouteFuncwTransformer(
                c_in=out_channels,
                ratio=rf_r,
                kernels=rf_k,
                with_bias_cal=bias,
                zero_init_cal=False,
                head_dim=head_dim
            )

        if internal_temp_aggr:
            self.bn_a = nn.BatchNorm3d(out_channels)
            self.bn_b = nn.BatchNorm3d(out_channels)
            self.bn_b.skip_init=True
            self.bn_b.weight.data.zero_()
            self.bn_b.bias.data.zero_()

            self.avgpool = nn.AvgPool3d(kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0))

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        trunc_normal_(self.weight, std=.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            if hasattr(m, "skip_init") and m.skip_init:
                return
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat, reshape_required=True, alpha=None):
        """
        Args:
            feat (tensor): feature to perform convolution on.
            reshape_required (bool): True if intput feat is the shape of (L, N, C), 
                where N=B*T
        """
        if reshape_required:
            assert self.num_frames is not None
            h = w = int(math.sqrt(feat.shape[0]))
            # L, N, C -> H, W, B, T, C
            feat = feat.reshape(h,w,-1,self.num_frames,feat.shape[-1]).permute(2,4,3,0,1)

        # generate calibration factors
        if alpha is None:
            alpha = self.rf_func(feat)

        if isinstance(alpha, list):
            w_alpha, b_alpha = alpha[0], alpha[1]
        else:
            w_alpha = alpha
            b_alpha = None
            
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = feat.size()
        feat = feat.permute(0,2,1,3,4).reshape(1,-1,h,w)

        if self.cal_dim == "cin":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (w_alpha.permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in//self.groups, kh, kw)
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (w_alpha.permute(0,2,1,3,4).unsqueeze(3) * self.weight).reshape(-1, c_in//self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            if b_alpha is not None:
                # b_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C
                bias = (b_alpha.permute(0,2,1,3,4).squeeze() * self.bias).reshape(-1)
            else:
                bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            feat, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
            dilation=self.dilation[1:], groups=self.groups * b * t)


        output = output.view(b,t,c_out,h,w).permute(0,2,1,3,4)
        if hasattr(self, "bn_a") and hasattr(self, "bn_b"):
            output = self.bn_a(output) + self.bn_b(self.avgpool(output))
        if reshape_required:
            output = output.permute(3,4,0,2,1).reshape(h*w,b*t,c_out)

        return output
    
    def __repr__(self):
        return f"TAdaConv2dV2({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " +\
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"