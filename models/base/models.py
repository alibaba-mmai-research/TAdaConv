#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

import torch
import torch.nn as nn
from utils.registry import Registry
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import HEAD_REGISTRY

MODEL_REGISTRY = Registry("Model")

class BaseVideoModel(nn.Module):
    """
    Standard video model.
    The model is divided into the backbone and the head, where the backbone
    extracts features and the head performs classification.

    The backbones can be defined in model/base/backbone.py or anywhere else
    as long as the backbone is registered by the BACKBONE_REGISTRY.
    The heads can be defined in model/module_zoo/heads/ or anywhere else
    as long as the head is registered by the HEAD_REGISTRY.

    The registries automatically finds the registered modules and construct 
    the base video model.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseVideoModel, self).__init__()
        self.cfg = cfg
        
        # the backbone is created according to meta-architectures 
        # defined in models/base/backbone.py
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)

        # the head is created according to the heads 
        # defined in models/module_zoo/heads
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(BaseVideoModel, self).train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self

@MODEL_REGISTRY.register()
class MoSINet(BaseVideoModel):
    def __init__(self, cfg):
        super(MoSINet, self).__init__(cfg)
    
    def forward(self, x):
        if isinstance(x, dict):
            x_data = x["video"]
        else:
            x_data = x
        b, n, c, t, h, w = x_data.shape
        x_data = x_data.reshape(b*n, c, t, h, w)
        res, logits = super(MoSINet, self).forward(x_data)
        pred = {}
        if isinstance(res, dict):
            for k, v in res.items():
                pred[k] = v
        else:
            pred["move_joint"] = res
        return pred, logits