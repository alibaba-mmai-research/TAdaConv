#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" S3D/S3DG branch. """ 

import torch
import torch.nn as nn

from models.base.base_blocks import (
    BRANCH_REGISTRY, InceptionBaseConv3D
)

class InceptionBlock3D(nn.Module):
    """
    Element constructing the S3D/S3DG.
    See models/base/backbone.py L99-186.

    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """
    def __init__(self, cfg, in_planes, out_planes):
        super(InceptionBlock3D, self).__init__()

        _gating = cfg.VIDEO.BACKBONE.BRANCH.GATING

        assert len(out_planes) == 6
        assert isinstance(out_planes, list)

        [num_out_0_0a, 
        num_out_1_0a, num_out_1_0b,
        num_out_2_0a, num_out_2_0b, 
        num_out_3_0b] = out_planes

        self.branch0 = nn.Sequential(
            InceptionBaseConv3D(cfg, in_planes, num_out_0_0a, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            InceptionBaseConv3D(cfg, in_planes, num_out_1_0a, kernel_size=1, stride=1),
            BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, num_out_1_0a, num_out_1_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            InceptionBaseConv3D(cfg, in_planes, num_out_2_0a, kernel_size=1, stride=1),
            BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, num_out_2_0a, num_out_2_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            InceptionBaseConv3D(cfg, in_planes, num_out_3_0b, kernel_size=1, stride=1),
        )

        self.out_channels = sum([num_out_0_0a, num_out_1_0b, num_out_2_0b, num_out_3_0b])

        self.gating = _gating 
        if _gating:
            self.gating_b0 = SelfGating(num_out_0_0a)
            self.gating_b1 = SelfGating(num_out_1_0b)
            self.gating_b2 = SelfGating(num_out_2_0b)
            self.gating_b3 = SelfGating(num_out_3_0b)


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if self.gating:
            x0 = self.gating_b0(x0)
            x1 = self.gating_b1(x1)
            x2 = self.gating_b2(x2)
            x3 = self.gating_b3(x3)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out

class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G"""
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor
    
@BRANCH_REGISTRY.register()
class STConv3d(nn.Module):
    """
    Element constructing the S3D/S3DG.
    See models/base/backbone.py L99-186.

    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """
    def __init__(self,cfg,in_planes,out_planes,kernel_size,stride,padding=0):
        super(STConv3d, self).__init__()
        if isinstance(stride, tuple):
            t_stride = stride[0]
            stride = stride[-1]
        else: # int
            t_stride = stride
        
        self.bn_mmt = cfg.BN.MOMENTUM
        self.bn_eps = cfg.BN.EPS
        self._construct_branch(
            cfg,
            in_planes,
            out_planes,
            kernel_size,
            stride,
            t_stride,
            padding
        )

    def _construct_branch(
        self,
        cfg,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        t_stride,
        padding=0
    ):
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size),
                              stride=(1,stride,stride),padding=(0,padding,padding), bias=False)
        self.conv2 = nn.Conv3d(out_planes,out_planes,kernel_size=(kernel_size,1,1),
                               stride=(t_stride,1,1),padding=(padding,0,0), bias=False)

        self.bn1=nn.BatchNorm3d(out_planes, eps=self.bn_eps, momentum=self.bn_mmt)
        self.bn2=nn.BatchNorm3d(out_planes, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        
        # init
        self.conv1.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.conv2.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        return x


