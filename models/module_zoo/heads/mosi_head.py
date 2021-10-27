#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" MoSI heads. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY

@HEAD_REGISTRY.register()
class MoSIHeadOnlyX(BaseHead):
    """
    Head for predicting MoSI categories.
    This head performs prediction on only x-axis.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): the global config object. 
        """
        self.num_classes = (cfg.VIDEO.HEAD.NUM_CLASSES - 1) + 1 * (not cfg.PRETRAIN.ZERO_OUT)
        super(MoSIHeadOnlyX, self).__init__(cfg)
    
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

        self.out_x = nn.Linear(dim, self.num_classes, bias=True)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=4)
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
            x (dict): dictionary with keys "move_x", indicating the category
                prediction on the x-axis.
            logits (Tensor): global average pooled features.
        """
        out = {}
        x = self.global_avg_pool(x)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        
        out["move_x"] = self.out_x(x)

        if not self.training:
            out["move_x"] = self.activation(out["move_x"])
            out["move_x"] = out["move_x"].mean([1, 2, 3])
        
        out["move_x"] = out["move_x"].view(out["move_x"].shape[0], -1)
        return out, x.view(x.shape[0], -1)

@HEAD_REGISTRY.register()
class MoSIHeadOnlyY(BaseHead):
    """
    Head for predicting MoSI categories.
    This head performs prediction on only y-axis.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): the global config object. 
        """
        self.num_classes = (cfg.VIDEO.HEAD.NUM_CLASSES - 1) + 1 * (not cfg.PRETRAIN.ZERO_OUT)
        super(MoSIHeadOnlyY, self).__init__(cfg)
    
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

        self.out_x = nn.Linear(dim, self.num_classes, bias=True)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=4)
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
            x (dict): dictionary with keys "move_y", indicating the category
                prediction on the y-axis.
            logits (Tensor): global average pooled features.
        """
        out = {}
        x = self.global_avg_pool(x)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        
        out["move_y"] = self.out_x(x)

        if not self.training:
            out["move_y"] = self.activation(out["move_y"])
            out["move_y"] = out["move_y"].mean([1, 2, 3])
        
        out["move_y"] = out["move_y"].view(out["move_y"].shape[0], -1)
        return out, x.view(x.shape[0], -1)

@HEAD_REGISTRY.register()
class MoSIHeadJoint(BaseHead):
    """
    Head for predicting MoSI categories.
    This head performs prediction jointly on both axes.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): the global config object. 
        """
        if cfg.PRETRAIN.DECOUPLE:
            # decoupled (meaning only one axis would have speed),
            self.num_classes = len(cfg.PRETRAIN.DATA_MODE) * (cfg.VIDEO.HEAD.NUM_CLASSES - 1) + 1 * (not cfg.PRETRAIN.ZERO_OUT)
        else:
            self.num_classes = (cfg.VIDEO.HEAD.NUM_CLASSES) ** len(cfg.PRETRAIN.DATA_MODE)
        super(MoSIHeadJoint, self).__init__(cfg)
    
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

        self.out_joint = nn.Linear(dim, self.num_classes, bias=True)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=4)
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
            x (Tensor): joint prediction on both axes.
            logits (Tensor): global average pooled features.
        """
        x = self.global_avg_pool(x)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        
        out = self.out_joint(x)

        if not self.training:
            out = self.activation(out)
            out = out.mean([1, 2, 3])
        
        out = out.view(out.shape[0], -1)
        return out, x.view(x.shape[0], -1)