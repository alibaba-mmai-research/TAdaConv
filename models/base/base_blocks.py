#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Basic blocks. """

import os
import abc
import torch
import torch.nn as nn
from utils.registry import Registry
from models.utils.params import update_3d_conv_params

from torchvision.utils import make_grid, save_image

from einops import rearrange, repeat

from models.utils.init_helper import lecun_normal_, trunc_normal_, _init_transformer_weights

STEM_REGISTRY = Registry("Stem")
BRANCH_REGISTRY = Registry("Branch")
HEAD_REGISTRY = Registry("Head")

class BaseModule(nn.Module):
    """
    Constructs base module that contains basic visualization function and corresponding hooks.
    Note: The visualization function has only tested in the single GPU scenario.
        By default, the visualization is disabled.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseModule, self).__init__()
        self.cfg = cfg
        self.id = 0
        if self.cfg.VISUALIZATION.ENABLE and self.cfg.VISUALIZATION.FEATURE_MAPS.ENABLE:
            self.base_output_dir = self.cfg.VISUALIZATION.FEATURE_MAPS.BASE_OUTPUT_DIR
            self.register_forward_hook(self.visualize_features)
    
    def visualize_features(self, module, input, output_x):
        """
        Visualizes and saves the normalized output features for the module.
        """
        # feature normalization
        b,c,t,h,w = output_x.shape
        xmin, xmax = output_x.min(1).values.unsqueeze(1), output_x.max(1).values.unsqueeze(1)
        x_vis = ((output_x.detach() - xmin) / (xmax-xmin)).permute(0, 1, 3, 2, 4).reshape(b, c*h, t*w).detach().cpu().numpy()
        if hasattr(self, "stage_id"):
            stage_id = self.stage_id
            block_id = self.block_id
        else:
            stage_id = 0
            block_id = 0
        for i in range(b):
            if not os.path.exists(f'{self.base_output_dir}/{self.cfg.VISUALIZATION.NAME}/im_{self.id+i}/'):
                os.makedirs(f'{self.base_output_dir}/{self.cfg.VISUALIZATION.NAME}/im_{self.id+i}/')
            # plt.imsave(
            #     f'{self.base_output_dir}/{self.cfg.VISUALIZATION.NAME}/im_{self.id+i}/layer_{stage_id}_{block_id}_feature.jpg', 
            #     x_vis[i]
            # )
        self.id += b

class BaseBranch(BaseModule):
    """
    Constructs the base convolution branch for ResNet based approaches.
    """
    def __init__(self, cfg, block_idx, construct_branch=True):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
            construct_branch (bool):   whether or not to automatically construct the branch.
                In the cases that the branch is not automatically contructed, e.g., some extra
                parameters need to be specified before branch construction, the branch could be
                constructed by "self._construct_branch" function.
        """
        super(BaseBranch, self).__init__(cfg)
        self.cfg = cfg
        update_3d_conv_params(cfg, self, block_idx)
        if construct_branch:
            self._construct_branch()
    
    def _construct_branch(self):
        if self.transformation == 'simple_block':
            # for resnet with the number of layers lower than 34, simple block is constructed.
            self._construct_simple_block()
        elif self.transformation == 'bottleneck':
            # for resnet with the number of layers higher than 34, bottleneck is constructed.
            self._construct_bottleneck()
    
    @abc.abstractmethod
    def _construct_simple_block(self):
        return
    
    @abc.abstractmethod
    def _construct_bottleneck(self):
        return
    
    @abc.abstractmethod
    def forward(self, x):
        return

class Base3DBlock(nn.Module):
    """
    Constructs a base 3D block, composed of a shortcut and a conv branch.
    """
    def __init__(
        self,
        cfg,
        block_idx,
    ):
        """
        Args: 
            cfg         (Config): global config object. 
            block_idx   (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(Base3DBlock, self).__init__()
        
        self.cfg = cfg
        update_3d_conv_params(cfg, self, block_idx)

        self._construct_block(
            cfg             = cfg,
            block_idx       = block_idx,
        )
            
    def _construct_block(
        self,
        cfg,
        block_idx,
    ):
        if self.dim_in != self.num_filters or self.downsampling:
            # if the spatial size or the channel dimension is inconsistent for the input and the output 
            # of the block, a 1x1x1 3D conv need to be performed. 
            self.short_cut = nn.Conv3d(
                self.dim_in,
                self.num_filters,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                bias=False
            )
            self.short_cut_bn = nn.BatchNorm3d(
                self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt
            )
        self.conv_branch = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, block_idx)
        self.relu = nn.ReLU(inplace=True)
            
    def forward(self, x):
        short_cut = x
        if hasattr(self, "short_cut"):
            short_cut = self.short_cut_bn(self.short_cut(short_cut))
        
        x = self.relu(short_cut + self.conv_branch(x))
        return x

class Base3DResStage(nn.Module):
    """
    ResNet Stage containing several blocks.
    """
    def __init__(
        self,
        cfg,
        num_blocks,
        stage_idx,
    ):
        """
        Args:
            num_blocks (int): number of blocks contained in this res-stage.
            stage_idx  (int): the stage index of this res-stage.
        """
        super(Base3DResStage, self).__init__()
        self.cfg = cfg
        self.num_blocks = num_blocks       
        self._construct_stage(
            cfg                     = cfg,
            stage_idx               = stage_idx,
        )
        
    def _construct_stage(
        self,
        cfg,
        stage_idx,
    ):
        res_block = Base3DBlock(
            cfg                     = cfg,
            block_idx               = [stage_idx, 0],
        )
        self.add_module("res_{}".format(1), res_block)
        for i in range(self.num_blocks-1):
            res_block = Base3DBlock(
                cfg                 = cfg,
                block_idx           = [stage_idx, i+1],
            )
            self.add_module("res_{}".format(i+2), res_block)
        if cfg.VIDEO.BACKBONE.NONLOCAL.ENABLE and stage_idx+1 in cfg.VIDEO.BACKBONE.NONLOCAL.STAGES:
            non_local = BRANCH_REGISTRY.get('NonLocal')(
                cfg                 = cfg,
                block_idx           = [stage_idx, i+2]
            )
            self.add_module("nonlocal", non_local)

    def forward(self, x):

        # performs computation on the convolutions
        for i in range(self.num_blocks):
            res_block = getattr(self, "res_{}".format(i+1))
            x = res_block(x)

        # performs non-local operations if specified.
        if hasattr(self, "nonlocal"):
            non_local = getattr(self, "nonlocal")
            x = non_local(x)
        return x

class InceptionBaseConv3D(BaseModule):
    """
    Constructs basic inception 3D conv.
    Modified from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """
    def __init__(self, cfg, in_planes, out_planes, kernel_size, stride, padding=0):
        super(InceptionBaseConv3D, self).__init__(cfg)
        self.conv = nn.Conv3d(in_planes, out_planes, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

@STEM_REGISTRY.register()
class Base2DStem(BaseModule):
    """
    Constructs basic ResNet 2D Stem.
    A single 2D convolution is performed in the base 2D stem.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base2DStem, self).__init__(cfg)

        self.cfg = cfg

        # loading the config for downsampling
        _downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        _downsampling_temporal  = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        if _downsampling:
            _stride = [1, 2, 2]
        else:
            _stride = [1, 1, 1]
        
        self._construct_block(
            cfg             = cfg,
            dim_in          = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS,
            num_filters     = cfg.VIDEO.BACKBONE.NUM_FILTERS[0],
            kernel_sz       = cfg.VIDEO.BACKBONE.KERNEL_SIZE[0],
            stride          = _stride,
            bn_eps          = cfg.BN.EPS,
            bn_mmt          = cfg.BN.MOMENTUM
        )

    def _construct_block(
        self, 
        cfg,
        dim_in, 
        num_filters, 
        kernel_sz,
        stride,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        self.a = nn.Conv3d(
            dim_in,
            num_filters,
            kernel_size = [1, kernel_sz[1], kernel_sz[2]],
            stride      = [1, stride[1], stride[2]],
            padding     = [0, kernel_sz[1]//2, kernel_sz[2]//2],
            bias        = False
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x

@STEM_REGISTRY.register()
class Base3DStem(BaseModule):
    """
    Constructs basic ResNet 3D Stem.
    A single 3D convolution is performed in the base 3D stem.
    """
    def __init__(
        self, 
        cfg
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base3DStem, self).__init__(cfg)

        self.cfg = cfg

        # loading the config for downsampling
        _downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        _downsampling_temporal  = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        if _downsampling:
            if _downsampling_temporal:
                _stride = [2, 2, 2]
            else:
                _stride = [1, 2, 2]
        else:
            _stride = [1, 1, 1]
        
        self._construct_block(
            cfg             = cfg,
            dim_in          = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS,
            num_filters     = cfg.VIDEO.BACKBONE.NUM_FILTERS[0],
            kernel_sz       = cfg.VIDEO.BACKBONE.KERNEL_SIZE[0],
            stride          = _stride,
            bn_eps          = cfg.BN.EPS,
            bn_mmt          = cfg.BN.MOMENTUM
        )
        
    def _construct_block(
        self, 
        cfg,
        dim_in, 
        num_filters, 
        kernel_sz,
        stride,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        self.a = nn.Conv3d(
            dim_in,
            num_filters,
            kernel_size = kernel_sz,
            stride      = stride,
            padding     = [kernel_sz[0]//2, kernel_sz[1]//2, kernel_sz[2]//2],
            bias        = False
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x

@HEAD_REGISTRY.register()
class BaseHead(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseHead, self).__init__()
        self.cfg = cfg
        dim             = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(
            dim,
            num_classes,
            dropout_rate,
            activation_func
        )

    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        self.out = nn.Linear(dim, num_classes, bias=True)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )

    def forward(self, x):
        """
        Returns:
            x (Tensor): classification predictions.
            logits (Tensor): global average pooled features.
        """
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))    
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x
        out = self.out(out)

        if not self.training:
            out = self.activation(out)
        
        out = out.view(out.shape[0], -1)
        return out, x.view(x.shape[0], -1)

@HEAD_REGISTRY.register()
class BaseHeadx2(BaseHead):
    """
    Constructs two base heads in parallel.
    This is specifically for EPIC-KITCHENS dataset, where 'noun' and 'verb' class are predicted.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseHeadx2, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_func == "identity":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )
    
    def forward(self, x):
        """
        Returns:
            x (dict): dictionary of classification predictions,
                with keys "verb_class" and "noun_class" indicating
                the predictions on the verb and noun.
            logits (Tensor): global average pooled features.
        """
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))    

        if hasattr(self, "dropout"):
            out1 = self.dropout(x)
            out2 = out1
        else:
            out1 = x
            out2 = x

        out1 = self.linear1(out1)
        out2 = self.linear2(out2)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
