#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Builder for video models. """

import sys
import torch
import torch.nn as nn

import traceback

import utils.logging as logging

from models.base.models import BaseVideoModel, MODEL_REGISTRY
from models.utils.model_ema import ModelEmaV2

logger = logging.get_logger(__name__)

def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (Config): global config object that provides specifics to construct the model.
        gpu_id (Optional[int]): specify the gpu index to build model.
    Returns:
        model: constructed model
        model_ema: copied model for ema
    """
    # Construct the model
    if MODEL_REGISTRY.get(cfg.MODEL.NAME) == None:
        # attempt to find standard models
        model = BaseVideoModel(cfg)
    else:
        # if the model is explicitly defined,
        # it is directly constructed from the model pool
        model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)

    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        model = model.cuda(device=cur_device)
    
    model_ema = None
    if cfg.MODEL.EMA.ENABLE:
        model_ema = ModelEmaV2(model, decay=cfg.MODEL.EMA.DECAY)

    try:
        # convert batchnorm to be synchronized across 
        # different GPUs if needed
        sync_bn = cfg.BN.SYNC
        if sync_bn == True and cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    except:
        sync_bn = None

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS*cfg.NUM_SHARDS > 1:
        # Make model replica operate on the current device
        if cfg.PAI:
            # Support distributed training on the cluster
            model = torch.nn.parallel.DistributedDataParallel(
                module=model
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device
            )

    return model, model_ema