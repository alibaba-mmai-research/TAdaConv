#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""
Functions that handle saving and loading of checkpoints.
Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/checkpoint.py.
For the codes from the slowfast repo, the copy right belongs to
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import copy
import numpy as np
import os
import pickle
from collections import OrderedDict
import torch
import math

import utils.bucket as bu
import utils.distributed as du
import utils.logging as logging

from torch.hub import tqdm, load_state_dict_from_url as load_url

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if du.is_master_proc() and not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = os.listdir(d) if os.path.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cfg, cur_epoch):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cfg (Config): the global config object.
        cur_epoch (int): current number of epoch of the model.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 >= cfg.OPTIMIZER.MAX_EPOCH - 10 and cfg.PRETRAIN.ENABLE is False:
        return True
    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0 or cur_epoch + 1 == cfg.OPTIMIZER.MAX_EPOCH


def save_checkpoint(path_to_job, model, model_ema, optimizer, epoch, cfg, model_bucket=None):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (Config): the global config object.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) and not cfg.PAI:
        return
    # Ensure that the checkpoint dir exists.
    if not os.path.exists(get_checkpoint_dir(path_to_job)):
        os.mkdir(get_checkpoint_dir(path_to_job))
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS*cfg.NUM_SHARDS > 1 else model.state_dict()
    normalized_sd = sub_to_normal_bn(sd)

    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": normalized_sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    if model_ema is not None:
        checkpoint["model_ema_state"] = model_ema.module.state_dict() if cfg.NUM_GPUS*cfg.NUM_SHARDS > 1 else model_ema.state_dict()
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1)
    with open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)

    # Upload checkpoints
    if model_bucket is not None and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH, 
            path_to_checkpoint, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )
    return path_to_checkpoint


def inflate_weight(state_dict_2d, state_dict_3d):
    """
    Inflate 2D model weights in state_dict_2d to the 3D model weights in
    state_dict_3d. The details can be found in:
    Joao Carreira, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    Args:
        state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
        state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
    Returns:
        state_dict_inflated (OrderedDict): a dict of inflated parameters.
    """
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info(
                "Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = (
                v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            )
        elif v2d.shape == v3d.shape:
            v3d = v2d
        else:
            logger.info(
                "Unexpected {}: {} -|> {}: {}".format(
                    k, v2d.shape, k, v3d.shape
                )
            )
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated

def checkpoint_preprocess(cfg, model_state):
    """
    Preprocess the checkpoints for video vision transformers. Enabled in the cfg.
    It includes processing the positional embedding by repeating and super-resolution,
    and processing the embedding function from patch-based to tubelet based.
    Args:
        model_state: the model state dictionary for processing.
    """
    logger.info("Preprocessing given checkpoint.")
    
    if cfg.TRAIN.FINE_TUNE and cfg.TRAIN.CHECKPOINT_PRE_PROCESS.POP_HEAD:
        logger.info("Poping heads.")
        to_pops = []
        for k in model_state.keys():
            if "head" in k:
                print(k)
                to_pops.append(k)
        for to_pop in to_pops:
            model_state.pop(to_pop)
    
    if cfg.TRAIN.CHECKPOINT_PRE_PROCESS.POS_EMBED == "repeat":
        # expanding the positional embedding by repeating.
        logger.info("Repeating positional embedding.")
        _, n, c = model_state["backbone.pos_embd"].shape
        f = cfg.DATA.NUM_INPUT_FRAMES
        if hasattr(cfg.VIDEO.BACKBONE, "TUBELET_SIZE"):
            f = f // cfg.VIDEO.BACKBONE.TUBELET_SIZE
        cls_pos_embd = model_state["backbone.pos_embd"][0,0,:].reshape(1, 1, c)
        input_pos_embd = model_state["backbone.pos_embd"][0,1:,:]
        input_pos_embd = input_pos_embd.unsqueeze(1).repeat((1,f,1,1)).reshape(1, f*(n-1), c)
        model_state["backbone.pos_embd"] = torch.cat((cls_pos_embd, input_pos_embd), dim=1)

    elif cfg.TRAIN.CHECKPOINT_PRE_PROCESS.POS_EMBED == "super-resolution":
        # when fine-tuning on a different resolution, super-resolution is needed 
        # on the positional embedding.
        logger.info("Super-resolution on positional embedding.")
        _, n, c = model_state["backbone.pos_embd"].shape
        cls_pos_embd = model_state["backbone.pos_embd"][0,0:1,:].reshape(1, 1, c)
        pos_embd = model_state["backbone.pos_embd"][0,1:,:]

        model_patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE
        train_crop_size = cfg.DATA.TRAIN_CROP_SIZE
        num_patches_per_side = train_crop_size // model_patch_size
        num_patches_per_side_ckp = int(math.sqrt(n-1))

        if num_patches_per_side != num_patches_per_side_ckp:
            # different spatial resolutions.
            logger.info("Performing super-resolution on positional embeddings.")
            pos_embd = pos_embd.reshape(1, num_patches_per_side_ckp, num_patches_per_side_ckp, -1)
            pos_embd = torch.nn.functional.interpolate(
                pos_embd.permute(0,3,1,2), size=(num_patches_per_side,num_patches_per_side), mode="bilinear"
            ).permute(0,2,3,1).reshape(1, num_patches_per_side**2, -1)
            model_state["backbone.pos_embd"] = torch.cat((cls_pos_embd, pos_embd), dim=1)
        if "backbone.temp_embd" in model_state.keys():
            # different temporal resolutions.
            logger.info("Performing super-resolution on temporal embeddings.")
            cls_temp_embd = model_state["backbone.temp_embd"][0,0:1,:].reshape(1, 1, c)
            temp_embd = model_state["backbone.temp_embd"][0,1:,:].unsqueeze(0)
            temporal_patch_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE
            num_frames = cfg.DATA.NUM_INPUT_FRAMES
            num_patches_temporal = num_frames // temporal_patch_size
            num_patches_temporal_ckp = temp_embd.shape[1]
            if num_patches_temporal != num_patches_temporal_ckp:
                temp_embd = torch.nn.functional.interpolate(
                    temp_embd.permute(0,2,1), size=(num_patches_temporal), mode="linear"
                ).permute(0,2,1)
                model_state["backbone.temp_embd"] = torch.cat((cls_temp_embd, temp_embd), dim=1)
    elif cfg.TRAIN.CHECKPOINT_PRE_PROCESS.POS_EMBED is None:
        logger.info("No process on positional embedding.")
    else:
        raise NotImplementedError

    if cfg.TRAIN.CHECKPOINT_PRE_PROCESS.PATCH_EMBD == "central_frame":
        # the central frame initialization for tublet embedding
        logger.info("Central frame tubelet embedding.")
        w = torch.zeros_like(model_state["backbone.stem.conv1.weight"]).repeat(1, 1, cfg.VIDEO.BACKBONE.TUBELET_SIZE, 1, 1)
        w[:,:,cfg.VIDEO.BACKBONE.TUBELET_SIZE//2,:,:] = model_state["backbone.stem.conv1.weight"].squeeze()
        model_state["backbone.stem.conv1.weight"] = w
    elif cfg.TRAIN.CHECKPOINT_PRE_PROCESS.PATCH_EMBD == "average":
        # the average initialization for tublet embedding
        logger.info("Averaging tubelet embedding.")
        w = model_state["backbone.stem.conv1.weight"].repeat(1, 1, cfg.VIDEO.BACKBONE.TUBELET_SIZE, 1, 1)
        w = w / float(cfg.VIDEO.BACKBONE.TUBELET_SIZE)
        model_state["backbone.stem.conv1.weight"] = w
    elif cfg.TRAIN.CHECKPOINT_PRE_PROCESS.PATCH_EMBD is None:
        logger.info("No process on patch/tubelet embedding.")
        pass
    else:
        raise NotImplementedError

    return model_state


def load_checkpoint(
    cfg,
    path_to_checkpoint,
    model,
    model_ema,
    data_parallel=True,
    optimizer=None,
    inflation=False,
    pre_process=False,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        inflation (bool): if True, inflate the weights from the checkpoint.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model
    if model_ema is not None:
        ms_ema = model_ema.module if data_parallel else model_ema
    if path_to_checkpoint[:5] == 'https':
        checkpoint = load_url(path_to_checkpoint)
        checkpoint = convert_imagenet_weights(cfg, checkpoint, ms)
    else:
        assert os.path.exists(
            path_to_checkpoint
        ), "Checkpoint '{}' not found".format(path_to_checkpoint)
        # Load the checkpoint on CPU to avoid GPU mem spike.
        with open(path_to_checkpoint, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
    model_state_dict_3d = (
        model.module.state_dict() if data_parallel else model.state_dict()
    )
    checkpoint["model_state"] = normal_to_sub_bn(
        checkpoint["model_state"], model_state_dict_3d
    )
    if pre_process:
        checkpoint_preprocess(cfg, checkpoint["model_state"])
        if "model_ema_state" in checkpoint.keys():
            checkpoint_preprocess(cfg, checkpoint["model_ema_state"])
    if inflation:
        # Try to inflate the model.
        inflated_model_dict = inflate_weight(
            checkpoint["model_state"], model_state_dict_3d
        )
        mismatch = ms.load_state_dict(inflated_model_dict, strict=False)
        logger.info("Keys in model not matched: {}".format(mismatch[0]))
        logger.info("Keys in checkpoint not matched: {}".format(mismatch[1]))
    else:
        mismatch = ms.load_state_dict(checkpoint["model_state"], strict=False)
        logger.info("Keys in model not matched: {}".format(mismatch[0]))
        logger.info("Keys in checkpoint not matched: {}".format(mismatch[1]))
        if "model_ema_state" in checkpoint.keys() and model_ema is not None:
            logger.info("Loading model ema weights.")
            ms_ema.load_state_dict(checkpoint["model_ema_state"], strict=False)
            logger.info("Keys in model not matched: {}".format(mismatch[0]))
            logger.info("Keys in checkpoint not matched: {}".format(mismatch[1]))
        else:
            if not "model_ema_state" in checkpoint.keys():
                logger.info("Model ema weights not loaded because no ema state stored in checkpoint.")
        # Load the optimizer state (commonly not done when fine-tuning)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint.keys():
        epoch = checkpoint["epoch"]
    else:
        epoch = -1
    return epoch


def sub_to_normal_bn(sd):
    """
    Convert the Sub-BN paprameters to normal BN parameters in a state dict.
    There are two copies of BN layers in a Sub-BN implementation: `bn.bn` and
    `bn.split_bn`. `bn.split_bn` is used during training and
    "compute_precise_bn". Before saving or evaluation, its stats are copied to
    `bn.bn`. We rename `bn.bn` to `bn` and store it to be consistent with normal
    BN layers.
    Args:
        sd (OrderedDict): a dict of parameters whitch might contain Sub-BN
        parameters.
    Returns:
        new_sd (OrderedDict): a dict with Sub-BN parameters reshaped to
        normal parameters.
    """
    new_sd = copy.deepcopy(sd)
    modifications = [
        ("bn.bn.running_mean", "bn.running_mean"),
        ("bn.bn.running_var", "bn.running_var"),
        ("bn.split_bn.num_batches_tracked", "bn.num_batches_tracked"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    for key in sd:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                new_sd[new_key] = new_sd.pop(key)

        for rm in to_remove:
            if rm in key and key in new_sd:
                del new_sd[key]

    for key in new_sd:
        if key.endswith("bn.weight") or key.endswith("bn.bias"):
            if len(new_sd[key].size()) == 4:
                assert all(d == 1 for d in new_sd[key].size()[1:])
                new_sd[key] = new_sd[key][:, 0, 0, 0]

    return new_sd


def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key


def normal_to_sub_bn(checkpoint_sd, model_sd):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        checkpoint_sd (OrderedDict): source dict of parameters.
        model_sd (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            if "bn.split_bn." in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape
            c2_blob_shape = checkpoint_sd[key].shape

            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = torch.cat(
                    [checkpoint_sd[key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
                logger.info(
                    "{} {} -> {}".format(
                        key, before_shape, checkpoint_sd[key].shape
                    )
                )
    return checkpoint_sd


def load_test_checkpoint(cfg, model, model_ema, model_bucket=None):
    """
    Loading checkpoint logic for testing.
    """
    read_from_oss = False
    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "" and cfg.TEST.CHECKPOINT_FILE_PATH is not None:
        # If no checkpoint found in MODEL_VIS.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TEST.CHECKPOINT_FILE_PATH and test it.
        _checkpoint_file_path = cfg.TEST.CHECKPOINT_FILE_PATH
        if _checkpoint_file_path.split(':')[0] == 'oss':
            model_bucket_name = _checkpoint_file_path.split('/')[2]
            if model_bucket is None or model_bucket.bucket_name != model_bucket_name:
                model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
            checkpoint_path = 'ckp{}.pyth'.format(cfg.LOCAL_RANK if hasattr(cfg, 'LOCAL_RANK') else 0)
            read_from_oss = bu.read_from_bucket(
                model_bucket,
                _checkpoint_file_path,
                checkpoint_path,
                model_bucket_name
            )
        else:
            checkpoint_path = cfg.TEST.CHECKPOINT_FILE_PATH
        logger.info("Load from given checkpoint file.\nCheckpoint file path: {}".format(_checkpoint_file_path))
        load_checkpoint(
            cfg,
            checkpoint_path,
            model,
            model_ema,
            cfg.NUM_GPUS*cfg.NUM_SHARDS > 1,
            optimizer=None,
            inflation=False,
            pre_process=False
        )
        if checkpoint_path == 'ckp.pyth':
            bu.clear_tmp_file(checkpoint_path)
    elif has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        load_checkpoint(cfg, last_checkpoint, model, model_ema, cfg.NUM_GPUS*cfg.NUM_SHARDS > 1, optimizer=None, inflation=False, pre_process=False)
        logger.info("Load from the last checkpoint file: {}".format(last_checkpoint))
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "" and cfg.TRAIN.CHECKPOINT_FILE_PATH is not None:
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        _checkpoint_file_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
        if _checkpoint_file_path.split(':')[0] == 'oss':
            model_bucket_name = _checkpoint_file_path.split('/')[2]
            if model_bucket is None or model_bucket.bucket_name != model_bucket_name:
                model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
            checkpoint_path = 'ckp{}.pyth'.format(cfg.LOCAL_RANK if hasattr(cfg, 'LOCAL_RANK') else 0)
            read_from_oss = bu.read_from_bucket(
                model_bucket,
                _checkpoint_file_path,
                checkpoint_path,
                model_bucket_name
            )
        else:
            checkpoint_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
        logger.info("Load from given checkpoint file.\nCheckpoint file path: {}".format(_checkpoint_file_path))
        load_checkpoint(
            cfg,
            checkpoint_path,
            model,
            model_ema,
            cfg.NUM_GPUS*cfg.NUM_SHARDS > 1,
            optimizer=None,
            inflation=False,
            pre_process=False
        )
        if checkpoint_path == 'ckp.pyth':
            bu.clear_tmp_file(checkpoint_path)
    else:
        logger.info(
            "Unknown way of loading checkpoint. Using with random initialization, only for debugging."
        )
    if read_from_oss:
        bu.clear_tmp_file(checkpoint_path)


def load_train_checkpoint(cfg, model, model_ema, optimizer, model_bucket=None):
    """
    Loading checkpoint logic for training.
    """
    read_from_oss = False
    if cfg.TRAIN.AUTO_RESUME and has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        logger.info("Load from last checkpoint, {}.".format(last_checkpoint))
        checkpoint_epoch = load_checkpoint(
            cfg, last_checkpoint, model, model_ema, cfg.NUM_GPUS*cfg.NUM_SHARDS > 1, optimizer, 
            pre_process=cfg.TRAIN.CHECKPOINT_PRE_PROCESS.ENABLE
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "" and cfg.TRAIN.CHECKPOINT_FILE_PATH is not None:
        _checkpoint_file_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
        if _checkpoint_file_path.split(':')[0] == 'oss':
            model_bucket_name = _checkpoint_file_path.split('/')[2]
            if model_bucket is None or model_bucket.bucket_name != model_bucket_name:
                model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
            checkpoint_path = 'ckp{}.pyth'.format(cfg.LOCAL_RANK if cfg.NUM_GPUS*cfg.NUM_SHARDS > 1 else 0)
            read_from_oss = bu.read_from_bucket(
                model_bucket,
                _checkpoint_file_path,
                checkpoint_path,
                model_bucket_name
            )
        else:
            checkpoint_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
        logger.info("Load from given checkpoint file.\nCheckpoint file path: {}".format(_checkpoint_file_path))
        checkpoint_epoch = load_checkpoint(
            cfg,
            checkpoint_path,
            model,
            model_ema,
            cfg.NUM_GPUS*cfg.NUM_SHARDS > 1,
            optimizer=None if cfg.TRAIN.FINE_TUNE else optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            pre_process=cfg.TRAIN.CHECKPOINT_PRE_PROCESS.ENABLE
        )
        start_epoch = 0 if cfg.TRAIN.FINE_TUNE else (checkpoint_epoch + 1)
        if read_from_oss:
            bu.clear_tmp_file(checkpoint_path)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH == "" and cfg.TRAIN.IMAGENET_INIT:
        if cfg.VIDEO.BACKBONE.META_ARCH == "ResNet3D":
            model_urls = {
                'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            }
            ckp_to_download = f"resnet{cfg.VIDEO.BACKBONE.DEPTH}"
            checkpoint_file_path = model_urls[ckp_to_download]
            logger.info("Load from imagenet pretrain.\nCheckpoint file path: {}".format(checkpoint_file_path))
            checkpoint_epoch = load_checkpoint(
                cfg,
                checkpoint_file_path,
                model,
                model_ema,
                cfg.NUM_GPUS*cfg.NUM_SHARDS > 1,
                optimizer=None if cfg.TRAIN.FINE_TUNE else optimizer,
                inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
                pre_process=cfg.TRAIN.CHECKPOINT_PRE_PROCESS.ENABLE
            )
        else:
            raise NotImplementedError
        start_epoch = 0 if cfg.TRAIN.FINE_TUNE else (checkpoint_epoch + 1)
    else:
        start_epoch = 0
    
    return start_epoch

def convert_imagenet_weights(cfg, src, tgt):
    tadaconv_enabled = cfg.VIDEO.BACKBONE.BRANCH.NAME in ["TAdaConvBlockAvgPool"]
    src_converted = {}
    for k, v in src.items():
        if 'fc' in k:
            continue
        if len(k.split('.')) == 2:
            mod, w_b = k.split('.')
            if mod == "conv1":
                # conv1.weight, torch.Size([64, 3, 7, 7]) 
                # -> 
                # backbone.conv1.a.weight, torch.Size([64, 3, 1, 7, 7])
                new_k = f'backbone.conv1.a.{w_b}'
                new_v = v.unsqueeze(2)
            elif mod == "bn1":
                # bn1.weight, torch.Size([64])
                # -> 
                # backbone.conv1.a_bn.weight, torch.Size([64])
                new_k = f'backbone.conv1.a_bn.{w_b}'
                new_v = v
        elif len(k.split('.')) == 4:
            layer, block_id, mod, w_b = k.split('.')
            layer_id = int(layer[-1]) + 1
            block_id = int(block_id) + 1
            mod_id = chr(ord('a') + int(mod[-1])-1)
            if "conv" in mod:
                # layer1.0.conv1.weight, torch.Size([64, 64, 1, 1])
                # ->
                # backbone.conv2.res_1.conv_branch.a.weight, torch.Size([64, 64, 1, 1, 1])
                new_k = f'backbone.conv{layer_id}.res_{block_id}.conv_branch.{mod_id}.{w_b}'
                if tadaconv_enabled and v.shape[-2:]!=(1,1):
                    new_v = v.unsqueeze(0).unsqueeze(0)
                else:
                    new_v = v.unsqueeze(2)
            elif "bn" in mod:
                # layer1.0.bn1.weight, torch.Size([64])
                # ->
                # backbone.conv2.res_1.conv_branch.a_bn.weight, torch.Size([64])
                new_k = f'backbone.conv{layer_id}.res_{block_id}.conv_branch.{mod_id}_bn.{w_b}'
                new_v = v
        elif len(k.split('.')) == 5:
            layer, block_id, _, mod, w_b = k.split('.')
            layer_id = int(layer[-1]) + 1
            block_id = int(block_id) + 1
            if mod == '0':
                # layer1.0.downsample.0.weight, torch.Size([256, 64, 1, 1])
                # -> 
                # backbone.conv2.res_1.short_cut.weight, torch.Size([256, 64, 1, 1, 1])
                new_k = f'backbone.conv{layer_id}.res_{block_id}.short_cut.{w_b}'
                new_v = v.unsqueeze(2)
            elif mod == '1':
                new_k = f'backbone.conv{layer_id}.res_{block_id}.short_cut_bn.{w_b}'
                new_v = v
        
        src_converted[new_k] = new_v
    
    # validate
    for k, v in src_converted.items():
        if k in tgt.state_dict().keys():
            if not tgt.state_dict()[k].shape == v.shape:
                logger.info(f"Size mismatch for converting from imagenet: should be {tgt.state_dict()[k].shape} for {k} instead of {v.shape}")
        else:
            logger.info(f"Didn't match any keys for {k}")
    print("")
    return {'model_state': src_converted}