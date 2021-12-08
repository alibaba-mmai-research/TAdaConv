#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Backbone/Meta architectures. """

import torch
import torch.nn as nn
import torchvision
from utils.registry import Registry
from models.base.base_blocks import (
    Base3DResStage, STEM_REGISTRY, BRANCH_REGISTRY, InceptionBaseConv3D
)
from models.module_zoo.branches.s3dg_branch import InceptionBlock3D
from models.utils.init_helper import _init_convnet_weights

BACKBONE_REGISTRY = Registry("Backbone")

_n_conv_resnet = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}

@BACKBONE_REGISTRY.register()
class ResNet3D(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(ResNet3D, self).__init__()
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        self.conv1 = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg=cfg)

        (n1, n2, n3, n4) = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]

        # ------------------- Main arch -------------------
        self.conv2 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n1,
            stage_idx               = 1,
        )

        self.conv3 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n2,
            stage_idx               = 2,
        )

        self.conv4 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n3,
            stage_idx               = 3,
        )

        self.conv5 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n4,
            stage_idx               = 4,
        )
        
        # perform initialization
        if cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming":
            _init_convnet_weights(self)
    
    def forward(self, x):
        if type(x) is list:
            x = x[0]
        elif isinstance(x, dict):
            x = x["video"]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

@BACKBONE_REGISTRY.register()
class Inception3D(nn.Module):
    """
    Backbone architecture for I3D/S3DG. 
    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Inception3D, self).__init__()
        _input_channel = cfg.DATA.NUM_INPUT_CHANNELS
        self._construct_backbone(
            cfg,
            _input_channel
        )

    def _construct_backbone(
        self, 
        cfg,
        input_channel
    ):
        # ------------------- Block 1 -------------------
        self.Conv_1a = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(
            cfg, input_channel, 64, kernel_size=7, stride=2, padding=3
        )

        self.block1 = nn.Sequential(self.Conv_1a) # (64, 32, 112, 112)

        # ------------------- Block 2 -------------------
        self.MaxPool_2a = nn.MaxPool3d(
            kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)
        ) 
        self.Conv_2b = InceptionBaseConv3D(cfg, 64, 64, kernel_size=1, stride=1) 
        self.Conv_2c = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(
            cfg, 64, 192, kernel_size=3, stride=1, padding=1
        ) 

        self.block2 = nn.Sequential(
            self.MaxPool_2a, # (64, 32, 56, 56)
            self.Conv_2b,    # (64, 32, 56, 56)
            self.Conv_2c)    # (192, 32, 56, 56)

        # ------------------- Block 3 -------------------
        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Mixed_3b = InceptionBlock3D(cfg, in_planes=192, out_planes=[64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionBlock3D(cfg, in_planes=256, out_planes=[128, 128, 192, 32, 96, 64])

        self.block3 = nn.Sequential(
            self.MaxPool_3a,    # (192, 32, 28, 28)
            self.Mixed_3b,      # (256, 32, 28, 28)
            self.Mixed_3c)      # (480, 32, 28, 28)

        # ------------------- Block 4 -------------------
        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = InceptionBlock3D(cfg, in_planes=480, out_planes=[192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionBlock3D(cfg, in_planes=512, out_planes=[160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionBlock3D(cfg, in_planes=512, out_planes=[128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionBlock3D(cfg, in_planes=512, out_planes=[112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionBlock3D(cfg, in_planes=528, out_planes=[256, 160, 320, 32, 128, 128])

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,    # (512, 16, 14, 14)
            self.Mixed_4c,    # (512, 16, 14, 14)
            self.Mixed_4d,    # (512, 16, 14, 14)
            self.Mixed_4e,    # (528, 16, 14, 14)
            self.Mixed_4f)    # (832, 16, 14, 14)

        # ------------------- Block 5 -------------------
        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = InceptionBlock3D(cfg, in_planes=832, out_planes=[256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionBlock3D(cfg, in_planes=832, out_planes=[384, 192, 384, 48, 128, 128])

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,    # (832, 8, 7, 7)
            self.Mixed_5c)    # (1024, 8, 7, 7)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x 

@BACKBONE_REGISTRY.register()
class SimpleLocalizationConv(nn.Module):
    """
    Backbone architecture for temporal action localization, which only contains three simple convs.
    """
    def __init__(self, cfg):
        super(SimpleLocalizationConv, self).__init__()
        _input_channel = cfg.DATA.NUM_INPUT_CHANNELS
        self.hidden_dim_1d = cfg.VIDEO.DIM1D
        self.layer_num = cfg.VIDEO.BACKBONE_LAYER
        self.groups_num = cfg.VIDEO.BACKBONE_GROUPS_NUM
        self._construct_backbone(
            cfg,
            _input_channel
        )

    def _construct_backbone(
        self, 
        cfg,
        input_channel
    ):
        self.conv_list = [
            nn.Conv1d(input_channel, self.hidden_dim_1d, kernel_size=3, padding=1, groups=self.groups_num),
            nn.ReLU(inplace=True)]
        assert self.layer_num >= 1
        for ln in range(self.layer_num-1):
            self.conv_list.append(nn.Conv1d(self.hidden_dim_1d, 
                                            self.hidden_dim_1d,
                                            kernel_size=3, padding=1, groups=self.groups_num))
            self.conv_list.append(nn.ReLU(inplace=True))
        self.conv_layer = nn.Sequential(*self.conv_list)


    def forward(self, x):
        x['video'] = self.conv_layer(x['video'])
        return x
