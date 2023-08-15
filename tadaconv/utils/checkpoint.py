#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""
Functions that handle saving and loading of checkpoints.
Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/checkpoint.py.
For the codes from the slowfast repo, the copy right belongs to
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import os
import copy
import math
import torch
import urllib
import pickle
import hashlib
import numpy as np
from collections import OrderedDict

import tadaconv.utils.bucket as bu
import tadaconv.utils.distributed as du
import tadaconv.utils.logging as logging

from torch.hub import tqdm, load_state_dict_from_url as load_url

logger = logging.get_logger(__name__)

def _download_clip(url: str, root: str):
    """
        from https://github.com/openai/CLIP/blob/main/clip/clip.py
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

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
    pre_process=False,
):
    """
    Load the checkpoint from the given file. 
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model
    if model_ema is not None:
        ms_ema = model_ema.module if data_parallel else model_ema
    if path_to_checkpoint[:5] == 'https' and (cfg.TRAIN.INIT == "in1k" or cfg.TRAIN.INIT == "in22k"):
        checkpoint = load_url(path_to_checkpoint)
        if cfg.VIDEO.BACKBONE.META_ARCH == "ResNet3D":
            checkpoint = convert_resnet_weights(cfg, checkpoint, ms)
        elif cfg.VIDEO.BACKBONE.META_ARCH == "ConvNeXt":
            checkpoint = convert_convnext_weights(cfg, checkpoint, ms)
    elif path_to_checkpoint[:5] == 'https' and cfg.TRAIN.INIT == "clip":
        path_to_checkpoint = _download_clip(path_to_checkpoint, os.path.expanduser("~/.cache/clip"))
        checkpoint = torch.load(path_to_checkpoint, map_location="cpu").state_dict()
        checkpoint = convert_vit_clip_weights(cfg, checkpoint, ms)
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
    logger.info("Unmatched keys in model could due to new parameters introduced," + \
        "and unmatched keys in checkpoint might be caused by removing structures from the original model." + \
        "Both are normal.")
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
            pre_process=False
        )
        if checkpoint_path == 'ckp.pyth':
            bu.clear_tmp_file(checkpoint_path)
    elif has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        load_checkpoint(cfg, last_checkpoint, model, model_ema, cfg.NUM_GPUS*cfg.NUM_SHARDS > 1, optimizer=None, pre_process=False)
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
            pre_process=cfg.TRAIN.CHECKPOINT_PRE_PROCESS.ENABLE
        )
        start_epoch = 0 if cfg.TRAIN.FINE_TUNE else (checkpoint_epoch + 1)
        if read_from_oss:
            bu.clear_tmp_file(checkpoint_path)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH == "" and cfg.TRAIN.INIT != "":
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            'convnext-tiny-in1k': 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
            'convnext-small-in1k': 'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth',
            'convnext-small-in22k': 'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth',
            'convnext-base-in1k': 'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth',
            'convnext-base-in22k': 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth',
            'vit-b16': 'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
            'vit-l14': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'
        }
        if cfg.VIDEO.BACKBONE.META_ARCH == "ResNet3D" and cfg.TRAIN.INIT == "in1k":
            ckp_to_download = f"resnet{cfg.VIDEO.BACKBONE.DEPTH}"
        elif cfg.VIDEO.BACKBONE.META_ARCH == "ConvNeXt" and (cfg.TRAIN.INIT == "in1k" or cfg.TRAIN.INIT == "in22k"):
            model_size = cfg.MODEL.NAME.split('-')[-1].lower()
            ckp_to_download = f'convnext-{model_size}-{cfg.TRAIN.INIT}'
        elif cfg.VIDEO.BACKBONE.META_ARCH == "VisionTransformer" and cfg.TRAIN.INIT == "clip":
            model_size = cfg.MODEL.NAME.split('_')[-1].lower()
            ckp_to_download = f'vit-{model_size}'
        else:
            raise NotImplementedError
        checkpoint_file_path = model_urls[ckp_to_download]
        logger.info(f"Load from {cfg.TRAIN.INIT} pretrain.\nCheckpoint file path: {checkpoint_file_path}")
        checkpoint_epoch = load_checkpoint(
            cfg,
            checkpoint_file_path,
            model,
            model_ema,
            cfg.NUM_GPUS*cfg.NUM_SHARDS > 1,
            optimizer=None if cfg.TRAIN.FINE_TUNE else optimizer,
            pre_process=cfg.TRAIN.CHECKPOINT_PRE_PROCESS.ENABLE
        )
        start_epoch = 0 if cfg.TRAIN.FINE_TUNE else (checkpoint_epoch + 1)
    else:
        start_epoch = 0
    
    return start_epoch

def convert_resnet_weights(cfg, src, tgt):
    tadaconv_enabled = cfg.VIDEO.BACKBONE.BRANCH.NAME in ["TAda2DBlock"]
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
    return {'model_state': src_converted}

def convert_convnext_weights(cfg, src, tgt):
    tadaconv_enabled = cfg.VIDEO.BACKBONE.BRANCH.NAME in ["TAdaConvNeXtV2Block"]
    src_converted = {}
    for k, v in src['model'].items():
        if 'head' in k:
            continue
        new_k = f'backbone.{k}'
        if "downsample_layers.0.0.weight" in k:
            if cfg.VIDEO.BACKBONE.STEM.T_KERNEL_SIZE == 3:
                pad = torch.zeros(v.shape)
                new_v = torch.stack((pad, v, pad), dim=2)
            else:
                new_v = v.unsqueeze(2).repeat(1,1,T_KERNEL_SIZE,1,1)
        elif len(v.shape) > 2:
            if "dwconv.weight" in k and tadaconv_enabled:
                new_v = v.unsqueeze(0).unsqueeze(0)
            else:
                new_v = v.unsqueeze(2)
        elif tadaconv_enabled and "dwconv.bias" in k:
            new_v = v.unsqueeze(0).unsqueeze(0)
        else:
            new_v = v
        
        src_converted[new_k] = new_v
    
    # validate
    for k, v in src_converted.items():
        if k in tgt.state_dict().keys():
            if not tgt.state_dict()[k].shape == v.shape:
                logger.info(f"Size mismatch for converting from imagenet: should be {tgt.state_dict()[k].shape} for {k} instead of {v.shape}")
        else:
            logger.info(f"Didn't match any keys for {k}")
    return {'model_state': src_converted}

def convert_vit_clip_weights(cfg, src, tgt):
    src_converted = {}
    for k, v in src.items():
        new_k = k
        if "visual" in k:
            new_k = new_k.replace("visual", "backbone")
            new_k = new_k.replace("transformer.resblocks", "layers")
            if new_k == "backbone.conv1.weight" and cfg.VIDEO.BACKBONE.TUBLET_SIZE == 3:
                pad = torch.zeros_like(v.unsqueeze(2))
                v = torch.cat((pad, v.unsqueeze(2), pad), dim=2)
        elif k in [
            "text_projection", "logit_scale"
        ]:
            pass   
        else:
            new_k = "text_backbone." + new_k
            new_k = new_k.replace("transformer.resblocks", "layers")
        src_converted[new_k] = v.to(torch.float32).cpu()

    # validate
    for k, v in src_converted.items():
        if "text" in k or "logit" in k:
            continue
        if k in tgt.state_dict().keys():
            if not tgt.state_dict()[k].shape == v.shape:
                logger.info(f"Size mismatch for converting from imagenet: should be {tgt.state_dict()[k].shape} for {k} instead of {v.shape}")
        else:
            logger.info(f"Didn't match any keys for {k}")
    return {'model_state': src_converted}