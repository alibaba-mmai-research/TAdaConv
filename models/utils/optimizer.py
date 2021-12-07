#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" 
Optimizer. 
Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/optimizer.py
For the codes from the slowfast repo, the copy right belongs to
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import torch

import utils.logging as logging
import utils.misc as misc
import models.utils.lr_policy as lr_policy
from models.utils.lars import LARS
import math

logger = logging.get_logger(__name__)


def construct_optimizer(model, cfg):
    """
    Construct an optimizer. 
    Supported optimizers include:
        SGD:    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
        ADAM:   Diederik P.Kingma, and Jimmy Ba. "Adam: A Method for Stochastic Optimization."
        ADAMW:  Ilya Loshchilov, and Frank Hutter. "Decoupled Weight Decay Regularization."
        LARS:   Yang You, Igor Gitman, and Boris Ginsburg. "Large Batch Training of Convolutional Networks."

    Args:
        model (model): model for optimization.
        cfg (Config): Config object that includes hyper-parameters for the optimizers. 
    """
    if cfg.TRAIN.ONLY_LINEAR:
        # only include linear layers
        params = []
        for name, p in model.named_parameters():
            if "head" in name:
                params.append(p)
        optim_params = [{"params": params, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY}]
    else:
        custom_parameters = []
        custom_bias_parameters = []
        custom_bn_parameters = []
        bn_parameters = []              # Batchnorm parameters.
        head_parameters = []            # Head parameters.
        head_bias_parameters = []       # Head bias parameters.
        non_bn_parameters = []          # Non-batchnorm parameters.
        non_bn_bias_parameters = []     # Non-batchnorm bias parameters.
        no_weight_decay_parameters = [] # No weight decay parameters.
        no_weight_decay_parameters_names = []
        num_skipped_param = 0
        for name, p in model.named_parameters():
            if hasattr(cfg.TRAIN, "FIXED_WEIGHTS") and (
                name.split('.')[1] in cfg.TRAIN.FIXED_WEIGHTS or 
                name.split('.')[2] in cfg.TRAIN.FIXED_WEIGHTS):
                # fixing weights to a certain extent
                logger.info("Fixed weight: {}".format(name))
                num_skipped_param += 1
                continue
            if "rf" in name:
                if "bn" in name:
                    custom_bn_parameters.append(p)
                elif "bias" in name:
                    custom_bias_parameters.append(p)
                else:
                    custom_parameters.append(p)
            elif "embd" in name or "cls_token" in name:
                no_weight_decay_parameters_names.append(name)
                no_weight_decay_parameters.append(p)
            elif "bn" in name or "norm" in name:
                bn_parameters.append(p)
            elif "head" in name:
                if "bias" in name:
                    head_bias_parameters.append(p)
                else:
                    head_parameters.append(p)
            else:
                if "bias" in name:
                    non_bn_bias_parameters.append(p)
                else:
                    non_bn_parameters.append(p)
        optim_params = [
            {"params": custom_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": 5 if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else 1},
            {"params": custom_bias_parameters, "weight_decay": 0.0, "lr_mult": 10 if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else (2 if cfg.OPTIMIZER.BIAS_DOUBLE else 1)},
            # normal params
            {"params": non_bn_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": 1},
            {"params": non_bn_bias_parameters, "weight_decay": 0.0, "lr_mult": 2 if cfg.OPTIMIZER.BIAS_DOUBLE else 1},
            # head params
            {"params": head_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": 5 if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else 1},
            {"params": head_bias_parameters, "weight_decay": 0.0, "lr_mult": 10 if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else (2 if cfg.OPTIMIZER.BIAS_DOUBLE else 1)},
            # no weight decay params
            {"params": no_weight_decay_parameters, "weight_decay": 0.0, "lr_mult": 1},
        ]
        if not cfg.BN.WB_LOCK:
            optim_params = [
                {"params": bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult": 1},
                {"params": custom_bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult": 1},
            ] + optim_params
        else:
            logger.info("Model bn/ln locked (not optimized).")

        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(custom_parameters) + \
            len(custom_bias_parameters) + \
            len(custom_bn_parameters) + \
            len(non_bn_parameters) + \
            len(non_bn_bias_parameters) + \
            len(bn_parameters) + \
            len(head_parameters) + \
            len(head_bias_parameters) + \
            len(no_weight_decay_parameters) + \
            num_skipped_param, "parameter size does not match: {} + {} != {}".format(len(non_bn_parameters), len(bn_parameters), len(list(model.parameters())))

        logger.info(f"Optimized parameters constructed. Parameters without weight decay: {no_weight_decay_parameters_names}")

    if cfg.OPTIMIZER.OPTIM_METHOD == "sgd":
        if cfg.OPTIMIZER.ADJUST_LR:
            # adjust learning rate for contrastive learning
            # the learning rate calculation is according to SimCLR
            num_clips_per_video = cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO if cfg.PRETRAIN.ENABLE else 1
            cfg.OPTIMIZER.BASE_LR = cfg.OPTIMIZER.BASE_LR*misc.get_num_gpus(cfg)*cfg.TRAIN.BATCH_SIZE*num_clips_per_video/256.
        return torch.optim.SGD(
            optim_params,
            lr=cfg.OPTIMIZER.BASE_LR,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            weight_decay=float(cfg.OPTIMIZER.WEIGHT_DECAY),
            dampening=cfg.OPTIMIZER.DAMPENING,
            nesterov=cfg.OPTIMIZER.NESTEROV,
        )
    elif cfg.OPTIMIZER.OPTIM_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.OPTIM_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.OPTIM_METHOD == "lars":
        if cfg.OPTIMIZER.ADJUST_LR:
            # adjust learning rate for contrastive learning
            # the learning rate calculation is according to SimCLR
            num_clips_per_video = cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO if cfg.PRETRAIN.ENABLE else 1
            cfg.OPTIMIZER.BASE_LR = cfg.OPTIMIZER.BASE_LR*misc.get_num_gpus(cfg)*cfg.TRAIN.BATCH_SIZE*num_clips_per_video/256.
        return LARS(
            optim_params,
            lr=cfg.OPTIMIZER.BASE_LR,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            weight_decay=float(cfg.OPTIMIZER.WEIGHT_DECAY),
            dampening=cfg.OPTIMIZER.DAMPENING,
            nesterov=cfg.OPTIMIZER.NESTEROV,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.OPTIMIZER.OPTIM_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cur_epoch (float): current poch id.
        cfg (Config): global config object, including the settings on 
            warm-up epochs, base lr, etc.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_idx, param_group in enumerate(optimizer.param_groups):
        if "lr_mult" in param_group.keys():
            # reduces the lr by a factor of 10 if specified for lr reduction
            param_group["lr"] = new_lr * param_group["lr_mult"]
        else:
            param_group["lr"] = new_lr
