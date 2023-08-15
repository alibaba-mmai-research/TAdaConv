#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Params. """

def update_3d_conv_params(cfg, conv, idx):
    """
    Automatically decodes parameters for 3D convolution blocks according to the config and its index in the model.
    Args: 
        cfg (Config):       Config object that contains model parameters such as channel dimensions, whether to downsampling or not, etc.
        conv (BaseBranch):  Branch whose parameters needs to be specified. 
        idx (list):         List containing the index of the current block. ([stage_id, block_id])
    """
    # extract current block location
    stage_id, block_id  = idx
    conv.stage_id       = stage_id
    conv.block_id       = block_id

    # extract basic info
    if block_id == 0:
        conv.dim_in                 = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id-1]
        if hasattr(cfg.VIDEO.BACKBONE, "ADD_FUSION_CHANNEL") and cfg.VIDEO.BACKBONE.ADD_FUSION_CHANNEL:
            conv.dim_in = conv.dim_in * cfg.VIDEO.BACKBONE.SLOWFAST.CONV_CHANNEL_RATIO // cfg.VIDEO.BACKBONE.SLOWFAST.BETA + conv.dim_in
        conv.downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[stage_id]
        conv.downsampling_temporal  = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[stage_id]
    else:
        conv.downsampling           = False
        conv.dim_in                 = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
    conv.num_filters                = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
    conv.bn_mmt                     = cfg.BN.MOMENTUM
    conv.bn_eps                     = cfg.BN.EPS
    conv.kernel_size                = cfg.VIDEO.BACKBONE.KERNEL_SIZE[stage_id]
    conv.expansion_ratio            = cfg.VIDEO.BACKBONE.EXPANSION_RATIO if hasattr(cfg.VIDEO.BACKBONE, "EXPANSION_RATIO") else None

    # configure downsampling
    if conv.downsampling:
        if conv.downsampling_temporal:
            conv.stride = [2, 2, 2]
        else:
            conv.stride = [1, 2, 2]
    else:
        conv.stride = [1, 1, 1]

    # define transformation
    if isinstance(cfg.VIDEO.BACKBONE.DEPTH, str):
        conv.transformation = 'bottleneck'
    else:
        if cfg.VIDEO.BACKBONE.DEPTH <= 34:
            conv.transformation = 'simple_block'
        else:
            conv.transformation = 'bottleneck'

    # calculate the input size
    num_downsampling_spatial = sum(
        cfg.VIDEO.BACKBONE.DOWNSAMPLING[:stage_id+(block_id>0)]
    )
    if 'DownSample' in cfg.VIDEO.BACKBONE.STEM.NAME:
        num_downsampling_spatial += 1
    num_downsampling_temporal = sum(
        cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[:stage_id+(block_id>0)]
    )
    conv.h = cfg.DATA.TRAIN_CROP_SIZE // 2**num_downsampling_spatial \
        + (cfg.DATA.TRAIN_CROP_SIZE//2**(num_downsampling_spatial-1))%2
    conv.w = conv.h
    conv.t = cfg.DATA.NUM_INPUT_FRAMES // 2**num_downsampling_temporal