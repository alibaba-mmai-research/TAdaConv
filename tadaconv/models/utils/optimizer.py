#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" 
Optimizer. 
Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/optimizer.py
For the codes from the slowfast repo, the copy right belongs to
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import json
import torch

import tadaconv.utils.logging as logging
import tadaconv.utils.misc as misc
import tadaconv.models.utils.lr_policy as lr_policy
from tadaconv.models.utils.lars import LARS
import math

logger = logging.get_logger(__name__)

def get_num_layer_for_vit(name, num_layers):
    if "module" in name:
        id_indicator_loc = 1
    else:
        id_indicator_loc = 0

    num_max_layer = num_layers + 2
    if "text" in name:
        return num_max_layer - 1
    elif "embd" in name or "cls_token" in name or "embedding" in name or "logit_scale" in name:
        return 0
    elif name.split('.')[id_indicator_loc] == 'backbone':
        if name.split('.')[id_indicator_loc+1] == "conv1" or name.split('.')[id_indicator_loc+1] == "ln_pre" :
            return 0
        elif name.split('.')[id_indicator_loc+1] == "ln_post" or name.split('.')[id_indicator_loc+1] == "proj":
            return num_max_layer - 1
        else:
            layer_id = int(name.split('.')[id_indicator_loc+2])
            return layer_id + 1
    else:
        return num_max_layer - 1

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
    elif cfg.OPTIMIZER.LAYER_WISE_LR_DECAY < 1.0:
        num_layers = cfg.VIDEO.BACKBONE.DEPTH
        get_num_layer = get_num_layer_for_vit
        if isinstance(num_layers, list):
            num_layers = 12
            get_num_layer = get_num_layer_for_convnext
        lr_mults = list(cfg.OPTIMIZER.LAYER_WISE_LR_DECAY ** (num_layers + 1 - i) for i in range(num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            if len(p.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                group_weight_decay = 0.
            else:
                group_name = "decay"
                group_weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY

            layer_id = get_num_layer(name, num_layers)
            group_name = f"layer_{layer_id}_{group_name}"
            
            is_new_param = False
            for param_name in cfg.OPTIMIZER.NEW_PARAMS:
                if param_name in name:
                    is_new_param = True
                    group_name = f"{group_name}_new_param"
                    break
            
            if cfg.OPTIMIZER.HEAD_LRMULT > 1.:
                if "head" in name:
                    group_name = f"{group_name}_head"
            
            if "text" in name:
                group_name = f"{group_name}_text"

            if group_name not in parameter_group_names:
                group_lr_mult = lr_mults[layer_id]
                
                if group_name.endswith("new_param"):
                    group_lr_mult *= cfg.OPTIMIZER.NEW_PARAMS_MULT
                if group_name.endswith("head"):
                    group_lr_mult *= cfg.OPTIMIZER.HEAD_LRMULT
                if group_name.endswith("text"):
                    group_lr_mult = cfg.OPTIMIZER.TEXT_LR_MULT if hasattr(cfg.OPTIMIZER, "TEXT_LR_MULT") else 0.1
                    if group_weight_decay > 0.:
                        group_weight_decay = cfg.OPTIMIZER.TEXT_WEIGHT_DECAY if hasattr(cfg.OPTIMIZER, "TEXT_WEIGHT_DECAY") else group_weight_decay

                parameter_group_names[group_name] = {
                    "params": [],
                    "weight_decay": group_weight_decay,
                    "lr_mult": group_lr_mult
                }
                parameter_group_vars[group_name] = {
                    "params": [],
                    "weight_decay": group_weight_decay,
                    "lr_mult": group_lr_mult
                }
            
            parameter_group_names[group_name]["params"].append(name)
            parameter_group_vars[group_name]["params"].append(p)
        
        logger.info(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
        optim_params = parameter_group_vars.values()
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
            if not p.requires_grad:
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
