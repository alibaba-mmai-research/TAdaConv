#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" SlowFast head. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseBranch, Base3DStem, BaseHead
from models.base.base_blocks import BRANCH_REGISTRY, HEAD_REGISTRY
from models.utils.init_helper import _init_convnet_weights

@HEAD_REGISTRY.register()
class SlowFastHead(nn.Module):
    """
    Constructs head for the SlowFast Networks. 
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(SlowFastHead, self).__init__()
        self.cfg = cfg
        self.mode = cfg.VIDEO.BACKBONE.SLOWFAST.MODE
        dim             = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        if self.mode == "slowfast":
            dim = dim + dim // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
        elif self.mode == "fastonly":
            dim = dim // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
        elif self.mode == "slowonly":
            pass
        else:
            raise NotImplementedError(
                "Mode {} not supported.".format(self.mode)
            )
        num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(
            dim,
            num_classes,
            dropout_rate,
            activation_func
        )
        _init_convnet_weights(self)

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
            x (Tensor): classification predictions.
            logits (Tensor): global average pooled features.
        """
        
        if self.mode == "slowfast":
            x = torch.cat(
                (self.global_avg_pool(x[0]), self.global_avg_pool(x[1])),
                dim=1
            )
        elif self.mode == "slowonly":
            x = self.global_avg_pool(x[0])
        elif self.mode == "fastonly":
            x = self.global_avg_pool(x[1])
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x
        out = self.out(out)

        if not self.training:
            out = self.activation(out)
            out = out.mean([1, 2, 3])
        
        out = out.view(out.shape[0], -1)
        return out, x.view(x.shape[0], -1)

@HEAD_REGISTRY.register()
class SlowFastHeadx2(nn.Module):
    """
    SlowFast Head for EPIC-KITCHENS dataset.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(SlowFastHeadx2, self).__init__()
        self.cfg = cfg
        self.mode = cfg.VIDEO.BACKBONE.SLOWFAST.MODE
        dim             = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        if self.mode == "slowfast":
            dim = dim + dim // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
        elif self.mode == "fastonly":
            dim = dim // cfg.VIDEO.BACKBONE.SLOWFAST.BETA
        elif self.mode == "slowonly":
            pass
        else:
            raise NotImplementedError(
                "Mode {} not supported.".format(self.mode)
            )
        num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(
            dim,
            num_classes,
            dropout_rate,
            activation_func
        )
        _init_convnet_weights(self)

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

        self.out1 = nn.Linear(dim, num_classes[0], bias=True)
        self.out2 = nn.Linear(dim, num_classes[1], bias=True)

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
            x (dict): dictionary of classification predictions,
                with keys "verb_class" and "noun_class" indicating
                the predictions on the verb and noun.
            logits (Tensor): global average pooled features.
        """
        
        if self.mode == "slowfast":
            x = torch.cat(
                (self.global_avg_pool(x[0]), self.global_avg_pool(x[1])),
                dim=1
            )
        elif self.mode == "slowonly":
            x = self.global_avg_pool(x[0])
        elif self.mode == "fastonly":
            x = self.global_avg_pool(x[1])
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        out1 = self.out1(x)
        out2 = self.out2(x)

        if not self.training:
            out1 = self.activation(out1)
            out1 = out1.mean([1, 2, 3])
            out2 = self.activation(out2)
            out2 = out2.mean([1, 2, 3])
        
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, x.view(x.shape[0], -1)