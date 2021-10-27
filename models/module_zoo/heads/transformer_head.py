#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Transformer heads. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY

from collections import OrderedDict
from models.utils.init_helper import lecun_normal_, trunc_normal_, _init_transformer_weights

@HEAD_REGISTRY.register()
class TransformerHead(BaseHead):
    """
    Construct head for video vision transformers.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(TransformerHead, self).__init__(cfg)
        self.apply(_init_transformer_weights)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        if self.cfg.VIDEO.HEAD.PRE_LOGITS:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
        
        self.linear = nn.Linear(dim, num_classes)

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
            x (Tensor): classification predictions.
            logits (Tensor): global average pooled features.
        """
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x
        if hasattr(self, "pre_logits"):
            out = self.pre_logits(out)
        out = self.linear(out)

        if not self.training:
            out = self.activation(out)
        return out, x

@HEAD_REGISTRY.register()
class TransformerHeadx2(BaseHead):
    """
    The Transformer head for EPIC-KITCHENS dataset.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(TransformerHeadx2, self).__init__(cfg)
        self.apply(_init_transformer_weights)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        if self.cfg.VIDEO.HEAD.PRE_LOGITS:
            self.pre_logits1 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
            self.pre_logits2 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
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
        if hasattr(self, "dropout"):
            out1 = self.dropout(x)
            out2 = self.dropout(x)
        else:
            out1 = x
            out2 = x

        if hasattr(self, "pre_logits1"):
            out1 = self.pre_logits1(out1)
            out2 = self.pre_logits2(out2)

        out1 = self.linear1(out1)
        out2 = self.linear2(out2)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        return {"verb_class": out1, "noun_class": out2}, x