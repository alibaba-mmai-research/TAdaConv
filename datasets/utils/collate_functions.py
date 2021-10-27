#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Collate functions. """

import random
from utils.registry import Registry
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F

COLLATE_FN_REGISTRY = Registry()

@COLLATE_FN_REGISTRY.register()
class ZeroShotCollate(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        batch = default_collate(batch) 
        batch[0]["text_embedding"] = batch[0]["text_embedding"][0].unsqueeze(0)
        return batch