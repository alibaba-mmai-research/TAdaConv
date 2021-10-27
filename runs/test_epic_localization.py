#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Test epic temporal action localization model by sliding windows."""

import numpy as np
import os
import pickle
import torch
import json
import math
import time

import utils.bucket as bu
import utils.checkpoint as cu
import utils.tensor as tu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
import utils.tal_tools as tal_tools
from datasets.base.builder import build_loader
from models.base.builder import build_model
from utils.meters import TestMeter
from utils.eval_tal.eval_tal import evaluate_detection

logger = logging.get_logger(__name__)


def fuse_epic_sliding_windows(maps_dict):
    """
        Since different sliding windows have overlap, we desigin this function to fuse sliding windows.
        Args:
            maps_dict (dict): {
                "start_map": (tensor)
                "end_map": (tensor)
                "verb_map": (tensor)
                "noun_map": (tensor)
                "confidence_map": (tensor)
                }
    """
    total_len = maps_dict['feat_mask_in_global'][-1][::2].shape[0]
    temporal_sum_count = torch.zeros(total_len)
    map_sum_count = torch.zeros(maps_dict['confidence_map'][0].size(1), total_len)
    sum_start = torch.zeros(total_len)
    sum_end = torch.zeros(total_len)
    sum_confidence_map = torch.zeros(maps_dict['confidence_map'][0].size(0), maps_dict['confidence_map'][0].size(1), total_len)
    sum_noun_map = torch.zeros(maps_dict['noun_map'][0].size(0), maps_dict['noun_map'][0].size(1), total_len)
    sum_verb_map = torch.zeros(maps_dict['verb_map'][0].size(0), maps_dict['verb_map'][0].size(1), total_len)
    for idx in range(len(maps_dict['feat_mask_in_global'])):
        mask = maps_dict['feat_mask_in_global'][idx][::2]
        mask = torch.cat([mask, torch.zeros(total_len-mask.size(0), dtype=torch.bool)])
        temporal_sum_count[mask] += 1
        sum_start[mask] += maps_dict['start'][idx][:mask.sum()]
        sum_end[mask] += maps_dict['end'][idx][:mask.sum()]
        sum_confidence_map[:, :, mask] += maps_dict['confidence_map'][idx][:, :, :mask.sum()]
        sum_noun_map[:, :, mask] += maps_dict['noun_map'][idx][:, :, :mask.sum()]
        sum_verb_map[:, :, mask] += maps_dict['verb_map'][idx][:, :, :mask.sum()]
        map_sum_count[:, mask] += maps_dict['map_mask'][:, :mask.sum()]
    temporal_sum_count[temporal_sum_count < 0.01] = 1.0
    map_sum_count[map_sum_count < 0.01] = 1.0
    results = {"start": sum_start / temporal_sum_count,
               "end": sum_end / temporal_sum_count,
               "confidence_map": sum_confidence_map / map_sum_count,
               "verb_map": sum_verb_map / map_sum_count,
               "noun_map": sum_noun_map / map_sum_count}
    return results


@torch.no_grad()
def perform_test(res_bucket, test_loader, model, test_meter, cfg, test_epoch, writer=None):
    """
    Perform sliding windows test on the specified test set
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (Config): The global config object.
    """
    model.eval()
    test_meter.iter_tic()
    res_dic = {}
    videos_name_list = []
    videos_map_dict = {}
    num_sample = 6
    cls_res_inter_matrix = model.module.head.get_interp1d_mask(0.0, num_sample) if hasattr(model, 'module') else model.head.get_interp1d_mask(0.0, num_sample)
    cls_res_inter_matrix = cls_res_inter_matrix.cuda()
    tscale = cfg.DATA.TEMPORAL_SCALE
    dscale = cfg.DATA.DURATION_SCALE if cfg.DATA.DURATION_SCALE > 0 else cfg.DATA.TEMPORAL_SCALE
    for cur_iter, (meta_dict, labels, indexes, meta) in enumerate(test_loader):
        meta_dict = tu.tensor2cuda(meta_dict)
        if cfg.DATA.LABELS_TYPE in ['bmn']:
            preds, logits = model(meta_dict)
        else:
            raise NotImplementedError(f"Unknown LABELS_TYPE:{cfg.DATA.LABELS_TYPE}")
        if cfg.DATA.LOAD_CLASSIFIER_RES:
            b, c, _ = meta_dict["verb_cls_data"].shape
            preds['verb_map'] = torch.matmul(meta_dict["verb_cls_data"], 
                                             cls_res_inter_matrix).reshape(b, c*num_sample, dscale, tscale)
            preds['verb_map'] = preds['verb_map'].reshape(b, 3, c//3, num_sample, dscale, tscale).softmax(dim=2).mean(dim=(1, 3))
            preds['verb_map'] = preds['verb_map'] * meta_dict['mask'][:, None, :, :]
            b, c, _ = meta_dict["noun_cls_data"].shape
            preds['noun_map'] = torch.matmul(meta_dict["noun_cls_data"], 
                                             cls_res_inter_matrix).reshape(b, c*num_sample, dscale, tscale)
            preds['noun_map'] = preds['noun_map'].reshape(b, 3, c//3, num_sample, dscale, tscale).softmax(dim=2).mean(dim=(1, 3))
            preds['noun_map'] = preds['noun_map'] * meta_dict['mask'][:, None, :, :]
        else:
            preds['noun_map'] = preds['noun_map'].softmax(dim=1) * meta_dict['mask'][:, None, :, :]
            preds['verb_map'] = preds['verb_map'].softmax(dim=1) * meta_dict['mask'][:, None, :, :]
        preds['confidence_map'] = preds['confidence_map'] * meta_dict['mask'][:, None, :, :]
        for b in range(len(meta_dict['video_name'])):
            video_name = meta_dict['video_name'][b]
            if video_name not in videos_map_dict:
                videos_map_dict[video_name] = {'confidence_map':[], 
                                               'start':[], 'end':[], 
                                               'noun_map':[], 'verb_map':[], 
                                               'feat_mask_in_global': [],
                                               'map_mask': meta_dict['mask'][0].cpu()}
                if len(videos_name_list) > 0:
                    # parse one bmn res
                    logger.info("fuse_epic_sliding_windows for video: {}".format(videos_name_list[-1]))
                    results_dict = fuse_epic_sliding_windows(videos_map_dict[videos_name_list[-1]])
                    logger.info("parse_epic_bmn_proposals for video: {}".format(videos_name_list[-1]))
                    new_props, heads = tal_tools.parse_epic_bmn_proposals(cfg, results_dict)
                    logger.info("save_epic_props for video: {}".format(videos_name_list[-1]))
                    tal_tools.save_epic_props(cfg, res_bucket, videos_name_list[-1], new_props, heads, cfg.LOCALIZATION.PROPS_DIR, test_epoch, to_oss=False)
                    videos_map_dict[videos_name_list[-1]] = {}
            videos_map_dict[video_name]['confidence_map'] += [preds['confidence_map'][b].cpu()]
            videos_map_dict[video_name]['start'] += [preds['start'][b].cpu()]
            videos_map_dict[video_name]['end'] += [preds['end'][b].cpu()]
            videos_map_dict[video_name]['noun_map'] += [preds['noun_map'][b].cpu()]
            videos_map_dict[video_name]['verb_map'] += [preds['verb_map'][b].cpu()]
            videos_map_dict[video_name]['feat_mask_in_global'] += [meta_dict['feat_mask_in_global'][b].cpu()]
            if video_name not in videos_name_list:
                videos_name_list.append(video_name)
        test_meter.iter_toc()
        test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()

    # for final video
    results_dict = fuse_epic_sliding_windows(videos_map_dict[videos_name_list[-1]])
    new_props, heads = tal_tools.parse_epic_bmn_proposals(cfg, results_dict)
    tal_tools.save_epic_props(cfg, res_bucket, videos_name_list[-1], new_props, heads, cfg.LOCALIZATION.PROPS_DIR, test_epoch, to_oss=False)
    test_meter.reset()
    return videos_name_list


def test_epic_localization(cfg):
    """
    Perform sliding windows testing on the pretrained video model.
    Args:
        cfg (CfgNode): Gobal configs.
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    cfg.TEST.BATCH_SIZE = 1
    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Test with config:")
        logger.info(cfg)


    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
    else:
        model_bucket = None
    cfg.TRAIN.ENABLE = False
    test_loader = build_loader(cfg, cfg.TEST.TEST_SET)
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    if du.get_world_size() != 1 or cfg.PAI is False or cfg.TEST.FORCE_FORWARD:
        # Build the video model and print model statistics.
        model, model_ema = build_model(cfg)
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        if len(cfg.TEST.TEST_CHECKPOINT) > 0:
            for test_epoch in cfg.TEST.TEST_CHECKPOINT:
                if test_epoch > 0:
                    path_to_checkpoint = cu.get_path_to_checkpoint(cfg.OUTPUT_DIR, test_epoch)
                    if os.path.exists(path_to_checkpoint) is True:
                        cfg.TEST.CHECKPOINT_FILE_PATH = path_to_checkpoint
                    elif model_bucket is not None:
                        cfg.TEST.CHECKPOINT_FILE_PATH = os.path.join(cfg.OSS.CHECKPOINT_OUTPUT_PATH, "checkpoint_epoch_{:05d}.pyth".format(test_epoch))
                    else:
                        cfg.TEST.CHECKPOINT_FILE_PATH = os.path.join(cfg.OUTPUT_DIR, "checkpoint_epoch_{:05d}.pyth".format(test_epoch))
                    logger.info("Testing model: {}".format(cfg.TEST.CHECKPOINT_FILE_PATH))
                    cu.load_test_checkpoint(cfg, model, model_ema, model_bucket)
                else:
                    assert len(cfg.TEST.TEST_CHECKPOINT) == 1 and test_epoch == -1
                # Create video testing loaders.
                cfg.LOG_PERIOD = 10
                test_meter = TestMeter(
                    cfg,
                    len(test_loader.dataset),
                    1,
                    1,
                    len(test_loader),
                )
                test_meter.set_model_ema_enabled(False)
                videos_name_list = perform_test(model_bucket, test_loader, model, test_meter, cfg, test_epoch)
                tal_tools.upload_results_to_oss(cfg, model_bucket, videos_name_list, test_epoch)
                if model_ema is not None:
                    test_meter.set_model_ema_enabled(True)
                    test_epoch = "{}_ema".format(test_epoch)
                    videos_name_list = perform_test(model_bucket, test_loader, model_ema.module, test_meter, cfg, test_epoch)
                    tal_tools.upload_results_to_oss(cfg, model_bucket, videos_name_list, test_epoch)

    if du.get_rank()==0 and cfg.TEST.TEST_SET not in 'training':
        for evaluate_epoch in cfg.TEST.TEST_CHECKPOINT:
            logger.info("Evaluating proposals for epoch {}".format(evaluate_epoch))
            result_path = tal_tools.epic_localization_post_processing(cfg, test_loader.dataset._video_name_duration, model_bucket, evaluate_epoch)
            if cfg.TEST.TEST_SET in 'validation':
                evaluate_detection(test_loader.dataset.local_anno_file, result_path, tiou_thresholds=np.array([0.1,0.2,0.3,0.4,0.5]))
                log_path = os.path.join(cfg.OUTPUT_DIR, "checkpoint_epoch_{}.pyth".format(evaluate_epoch))
                logger.info("Evaluating {} done with post processing {}".format(log_path, cfg.LOCALIZATION.POST_PROCESS.__dict__))

    if model_bucket is not None and cfg.TEST.TEST_SET in 'validation':
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE)
        if cfg.OSS.CHECKPOINT_OUTPUT_PATH[:3] == "oss":
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
            if cfg.TEST.SAVE_RESULTS_PATH != "":
                filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)
                bu.put_to_bucket(
                    model_bucket, 
                    cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                    filename,
                    cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
                )
                if hasattr(cfg.TEST, "RECORD_SSL_TEST") and cfg.TEST.RECORD_SSL_TEST:
                    filename = os.path.join(cfg.OUTPUT_DIR, "{}_ssl".format(cfg.TEST.SAVE_RESULTS_PATH))
                    bu.put_to_bucket(
                        model_bucket, 
                        cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                        filename,
                        cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
                    )
        else:
            log_dir = os.path.join(cfg.OSS.CHECKPOINT_OUTPUT_PATH, "log/")
            if os.path.exists(log_dir) is False:
                os.makedirs(log_dir)
            os.system("mv {} {}".format(filename, log_dir))