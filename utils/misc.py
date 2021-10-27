#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""
Miscellaneous.
Modifed from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/misc.py.
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import json
import logging
import math
import numpy as np
import os
from datetime import datetime
import psutil
import torch
from torch import nn

import utils.logging as logging

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def _get_model_analysis_input(cfg, use_train_input):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (Config): the global config object.
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.

    Returns:
        Args: the input for model analysis.
    """
    rgb_dimension = 3
    if use_train_input:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_INPUT_FRAMES,
            cfg.DATA.TRAIN_CROP_SIZE,
            cfg.DATA.TRAIN_CROP_SIZE,
        )
    else:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_INPUT_FRAMES,
            cfg.DATA.TEST_CROP_SIZE,
            cfg.DATA.TEST_CROP_SIZE,
        )
    model_inputs = input_tensors.unsqueeze(0)
    if cfg.NUM_GPUS:
        model_inputs = model_inputs.cuda(non_blocking=True)
    inputs = {"video": model_inputs}
    return inputs


def get_model_stats(model, cfg, mode, use_train_input):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (Config): the global config object.
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).
        use_train_input (bool): if True, compute statistics for training. Otherwise,
            compute statistics for testing.

    Returns:
        float: the total number of count of the given model.
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' not supported for model analysis".format(mode)
    try:
        from fvcore.nn.activation_count import activation_count
        from fvcore.nn.flop_count import flop_count
        if mode == "flop":
            model_stats_fun = flop_count
            from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS
            _DEFAULT_SUPPORTED_OPS["aten::batch_norm"] = None
        elif mode == "activation":
            model_stats_fun = activation_count
            from fvcore.nn.activation_count import _DEFAULT_SUPPORTED_OPS

        # Set model to evaluation mode for analysis.
        # Evaluation mode can avoid getting stuck with sync batchnorm.
        model_mode = model.training
        model.eval()
        inputs = _get_model_analysis_input(cfg, use_train_input)
        count_dict, _ = model_stats_fun(model, inputs, _DEFAULT_SUPPORTED_OPS)
        count = sum(count_dict.values())
        model.train(model_mode)
    except:
        count = None
    return count


def log_model_info(model, cfg, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage, gflops and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        cfg (Config): the global config object.
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    flops = get_model_stats(model, cfg, "flop", use_train_input)
    activations = get_model_stats(model, cfg, "activation", use_train_input)
    if flops is not None:
        logger.info("Flops: {:,} G".format(flops))
    if activations is not None:
        logger.info("Activations: {:,} M".format(activations))
    logger.info("nvidia-smi")
    os.system("nvidia-smi")


def is_eval_epoch(cfg, cur_epoch, multigrid_schedule=None):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (Config): the global config object.
        cur_epoch (int): current epoch.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cfg.TRAIN.EVAL_PERIOD == 0:
        return False
    if cur_epoch + 1 >= cfg.OPTIMIZER.MAX_EPOCH - 10 and (not cfg.PRETRAIN.ENABLE):
        return True
    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()

def get_num_gpus(cfg):
    if cfg.PAI:
        return cfg.NUM_GPUS * cfg.NUM_SHARDS
    else:
        return cfg.NUM_GPUS