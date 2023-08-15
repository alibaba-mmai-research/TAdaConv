#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Backbone/Meta architectures. """

import torch
import torch.nn as nn
import torchvision
from tadaconv.utils.registry import Registry
from tadaconv.models.base.base_blocks import (
    Base3DResStage, STEM_REGISTRY, BRANCH_REGISTRY, InceptionBaseConv3D
)
from tadaconv.models.module_zoo.ops.misc import LayerNorm
from tadaconv.models.utils.init_helper import trunc_normal_
from tadaconv.models.utils.init_helper import _init_convnet_weights
from tadaconv.models.module_zoo.branches.s3dg_branch import InceptionBlock3D

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

@BACKBONE_REGISTRY.register()
class VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        backbone_cfg = cfg.VIDEO.BACKBONE

        input_resolution    = backbone_cfg.INPUT_RES
        patch_size          = backbone_cfg.PATCH_SIZE
        tublet_size         = backbone_cfg.TUBLET_SIZE
        tublet_stride       = backbone_cfg.TUBLET_STRIDE
        width               = backbone_cfg.NUM_FEATURES
        depth               = backbone_cfg.DEPTH
        heads               = backbone_cfg.NUM_HEADS
        drop_path           = backbone_cfg.DROP_PATH
        output_dim          = cfg.VIDEO.HEAD.OUTPUT_DIM
        requrie_proj        = backbone_cfg.REQUIRE_PROJ

        self.num_patches_per_axis = input_resolution // patch_size
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            in_channels=3, 
            out_channels=width, 
            kernel_size=(tublet_size, patch_size, patch_size), 
            stride=(tublet_stride, patch_size, patch_size), 
            padding=(tublet_size//2, 0, 0),
            bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.ln_post = nn.LayerNorm(width)   

        if requrie_proj:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        else:
            self.proj = None
        
    def forward(self, x: torch.Tensor):
        if isinstance(x, dict):
            x = x["video"]
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        if len(x.shape) == 5:
            # means forwarding a batch of videos
            b, c, t, h, w = x.shape
            # x = x.permute(0,2,1,3,4).reshape(b*t, c, h, w)
            x = self.forward_wo_head(x)

            x = self.ln_post(x[:,0,:].reshape(b,-1,x.shape[-1]).mean(1))

            if self.proj is not None:
                x = x @ self.proj
        else:
            raise NotImplementedError

        return x
    
    def forward_wo_head(self, x: torch.Tensor):

        x = self.conv1(x)  # shape = [*, width, grid, grid]

        b,c,t,h,w = x.shape
        x = x.permute(0,2,3,4,1).reshape(b*t,h*w,c)

        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.layers(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x

@BACKBONE_REGISTRY.register()
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, cfg):
        super().__init__()

        in_chans = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS
        dims = cfg.VIDEO.BACKBONE.NUM_FILTERS
        drop_path_rate = cfg.VIDEO.BACKBONE.DROP_PATH
        depths = cfg.VIDEO.BACKBONE.DEPTH
        layer_scale_init_value = cfg.VIDEO.BACKBONE.LARGE_SCALE_INIT_VALUE
        stem_t_kernel_size = cfg.VIDEO.BACKBONE.STEM.T_KERNEL_SIZE
        t_stride = cfg.VIDEO.BACKBONE.STEM.T_STRIDE

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(stem_t_kernel_size,4,4), stride=(t_stride,4,4),padding=((stem_t_kernel_size-1)//2,0,0)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,2,2), stride=(1,2,2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(
                    cfg,
                    dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            if hasattr(m, "skip_init") and m.skip_init:
                return
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-3, -2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['video']
        x = self.forward_features(x)
        return x