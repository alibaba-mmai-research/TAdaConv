#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Temporal Action Localization tools for post-processing. """

import numpy as np
import utils.bucket as bu
import pandas as pd
import os, time, json
import torch
import math
from tqdm import tqdm
from utils import logging
import multiprocessing as mp
from utils.bboxes_1d import iou_with_anchors

logger = logging.get_logger(__name__)


def _get_oss_path_prefix(prefix, epoch, test_set):
    """
    Get full path with epoch and subset
    Args:
        prefix (str): path prefix.
        epoch (int): epoch number of these proposals
        test_set (str): training or validation set
    """
    return prefix + "_ep{}_{}".format(epoch, test_set)


def save_epic_props(cfg, bucket, video_name, propos, heads, dir_prefix, epoch, to_oss=False):
    """
    Save a video proposals for epic-kitchen.
    Args:
        cfg (Config): the global config object.
        bucket (Oss bucket): ignore it if you donot use oss
        video_name (str): video name
        propos (tensor): proposals for this video
        heads (list): proposals table head
        dir_prefix (str): path prefix
        epoch (int): epoch number of these proposals
    """
    # dir_prefix = cfg.LOCALIZATION.PROPS_DIR
    if bucket is not None and cfg.OSS.CHECKPOINT_OUTPUT_PATH[:3] == "oss":
        dir_name = os.path.join(cfg.LOCALIZATION.TEST_OUTPUT_DIR, dir_prefix)
    else:
        dir_name = os.path.join(cfg.OUTPUT_DIR, 
                                _get_oss_path_prefix(dir_prefix, epoch, cfg.TEST.TEST_SET))
        to_oss = False
    if os.path.exists(dir_name) is False:
        try:
            os.makedirs(dir_name)
        except:
            pass
    local_file = os.path.join(dir_name, video_name + ".pkl")
    logger.info("saving: {}".format(local_file))
    torch.save([propos, heads], local_file)
    logger.info("{} saved!!".format(local_file))
    if to_oss:
        oss_dir_prefix = _get_oss_path_prefix(dir_prefix, epoch, cfg.TEST.TEST_SET)
        oss_file = os.path.join(cfg.OSS.CHECKPOINT_OUTPUT_PATH, oss_dir_prefix + '/')
        bu.put_to_bucket(bucket, oss_file, local_file, bucket.bucket_name, retries=1, verbose=False)
    # bu.clear_tmp_file([local_file])
    return None


def parse_epic_bmn_proposals(cfg, results_dict):
    """
    Parse epic proposals by BMN map
    Args:
        cfg (Config): the global config object.
        results_dict (dict): Maps output by BMN.
    return:
        Proposals list parsed by this function and their table head.
    """
    start_scores = results_dict['start'].numpy()
    end_scores = results_dict['end'].numpy()
    clr_confidence = results_dict['confidence_map'][1].numpy()
    reg_confidence = results_dict['confidence_map'][0].numpy()
    verb_map = results_dict['verb_map'].numpy()
    noun_map = results_dict['noun_map'].numpy()
    reg_map = None
    if reg_map is not None:
        reg_map = reg_map.detach().cpu().numpy()
        weight_dx = cfg.VIDEO.HEAD.BMN_REG_DX_WEIGHT
        weight_dw = cfg.VIDEO.HEAD.BMN_REG_DW_WEIGHT

    tscale = clr_confidence.shape[1]
    dscale = clr_confidence.shape[0]
    max_start = max(start_scores)
    max_end = max(end_scores)

    ####################################################################################################
    # generate the set of start points and end points
    start_bins = np.zeros(len(start_scores))
    start_bins[0] = 1
    for idx in range(1, tscale - 1):
        if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
            start_bins[idx] = 1
        elif start_scores[idx] > (0.5 * max_start):
            start_bins[idx] = 1

    end_bins = np.zeros(len(end_scores))
    end_bins[-1] = 1
    for idx in range(1, tscale - 1):
        if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
            end_bins[idx] = 1
        elif end_scores[idx] > (0.5 * max_end):
            end_bins[idx] = 1
    ########################################################################################################

    #########################################################################
    new_props = []
    for idx in tqdm(range(dscale), total=dscale):
        for jdx in range(tscale):
            start_index = jdx
            end_index = start_index + idx+1
            if end_index < tscale and start_bins[start_index] == 1 and end_bins[end_index] == 1:
                xmin = start_index/tscale
                xmax = end_index/tscale
                xmin_score = start_scores[start_index]
                xmax_score = end_scores[end_index]
                clr_score = clr_confidence[idx, jdx]
                reg_score = reg_confidence[idx, jdx]
                verb_noun, vn_score = fuse_verb_noun_map(cfg, torch.Tensor(verb_map[:, idx, jdx]), torch.Tensor(noun_map[:, idx, jdx]))
                if reg_map is not None:
                    dx, dw = reg_map[0, idx, jdx] * weight_dx, reg_map[1, idx, jdx] * weight_dw
                    new_center = (xmax + xmin) / 2.0 + dx * (xmax - xmin)
                    new_width = (xmax - xmin) * np.exp(dw)
                    xmin, xmax = max(new_center - new_width / 2.0, 0), min(new_center + new_width / 2.0, 1.0)
                score = xmin_score * xmax_score * clr_score * reg_score
                new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score, verb_noun, vn_score])
    new_props = np.stack(new_props)
    #########################################################################
    heads = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_score", "score", "verb_noun", "vn_score"]
    return new_props, heads


def fuse_verb_noun_map(cfg, verb_map, noun_map):
    """
    Simply fuse verb map and noun map for action classification
    Args:
        cfg (Config): the global config object.
        verb_map (dict): verb classification for each proposal.
        noun_map (dict): noun classification for each proposal.
    return:
        Action classification list.
    """
    verb_topk, noun_topk = 10, 30
    verb_index = (-verb_map).argsort()
    sverb_map = verb_map[verb_index[:verb_topk]]
    noun_index = (-noun_map).argsort()
    snoun_map = noun_map[noun_index[:noun_topk]]
    fuse_map = sverb_map[None, :] * snoun_map[:, None]

    score_idx = (-fuse_map).reshape(-1).argsort()
    pesdo_noun_index = score_idx // verb_topk
    pesdo_verb_index = score_idx % verb_topk
    real_noun_index = noun_index[:noun_topk][pesdo_noun_index]
    real_verb_index = verb_index[:verb_topk][pesdo_verb_index]
    fuse_score = fuse_map.reshape(-1)[score_idx]
    verb_score = verb_map[real_verb_index]
    noun_score = noun_map[real_noun_index]
    topk = 20
    index = torch.stack([real_verb_index, real_noun_index], dim=1)[:topk, :].numpy()
    score = torch.stack([verb_score, noun_score, fuse_score], dim=1)[:topk, :].numpy()
    return index, score


def proposals_post_processing(cfg, video_list, epoch, post_func, norm_props=False):
    """
    Post processing for videos by multiprocessing.
    Args:
        cfg (Config): the global config object.
        video_list (list): videos name list.
        epoch (int): epoch number of these proposals
        post_func (func): post processing function for this dataset.
    return:
        Processed action localization results.
    """
    result_dict = mp.Manager().dict()
    num_videos = len(video_list)
    post_process_thread = cfg.LOCALIZATION.POST_PROCESS.THREAD
    group_video_list = [[] for i in range(post_process_thread)]
    for idx in range(num_videos):
        group_video_list[idx%post_process_thread].append(video_list[idx])
    processes = []
    for tid in range(post_process_thread - 1):
        tmp_video_list = group_video_list[tid]
        p = mp.Process(target=post_func, args=(cfg, tmp_video_list, result_dict, epoch, norm_props))
        p.start()
        processes.append(p)
    tmp_video_list = group_video_list[post_process_thread - 1]
    p = mp.Process(target=post_func, args=(cfg, tmp_video_list, result_dict, epoch, norm_props))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    return result_dict


def epic_localization_post_processing(cfg, video_list, bucket, epoch):
    """
    Post processing for epic dataset.
    Args:
        cfg (Config): the global config object.
        video_list (list): videos name list.
        bucket (Oss bucket): ignore it if you donot use oss
        epoch (int): epoch number of these proposals
    return:
        Processed action localization results for epic dataset.
    """
    video_list.sort(key=lambda x: x[1])
    post_process_thread = cfg.LOCALIZATION.POST_PROCESS.THREAD
    balanced_vide_list = []
    for idx in range(post_process_thread):
        balanced_vide_list.extend(video_list[idx::post_process_thread])
    assert len(balanced_vide_list) == len(video_list)
    result_dict = proposals_post_processing(cfg, balanced_vide_list, epoch, epic_video_post_process)

    if bucket is not None and cfg.OSS.CHECKPOINT_OUTPUT_PATH[:3] == "oss":
        result_path = _get_oss_path_prefix(cfg.LOCALIZATION.RESULT_FILE, epoch, cfg.TEST.TEST_SET) + '.json'
    else:
        result_path = os.path.join(cfg.OUTPUT_DIR,
                                   _get_oss_path_prefix(cfg.LOCALIZATION.RESULT_FILE, epoch, cfg.TEST.TEST_SET) + '.json')

    output_dict = {
        "version": "0.2",
        "challenge": "action_detection",
        "sls_pt": 2,
        "sls_tl": 3,
        "sls_td": 3,
        "results":result_dict}
    logger.info("epic post_processing done! saving epic detection results....")
    with open(result_path, "w") as f:
        json.dump(output_dict, f, indent=4)
    logger.info("epic detection results saved!")

    if bucket is not None and cfg.OSS.CHECKPOINT_OUTPUT_PATH[:3] == "oss":
        oss_path = cfg.OSS.CHECKPOINT_OUTPUT_PATH if cfg.OSS.CHECKPOINT_OUTPUT_PATH[-1:] == '/' else cfg.OSS.CHECKPOINT_OUTPUT_PATH + '/'
        logger.info("start uploading {}  file to oss!".format(result_path))
        bu.put_to_bucket(bucket, oss_path, result_path, bucket.bucket_name, retries=1)
        logger.info("uploaded {} to oss successful!".format(result_path))
    return result_path


def soft_nms(df, alpha, t1, t2, prop_num, iou_power=2.0):
    '''
    Soft nms for one video.
    Args:
        df: proposals generated by network;
        alpha: alpha value of Gaussian decaying function;
        t1, t2: threshold for soft nms.
    '''
    df = df.sort_values(by="score", ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    tindex = df.index.values[:].tolist()

    rstart = []
    rend = []
    rscore = []
    rindex = []

    while len(tscore) > 1 and len(rscore) < prop_num:
        max_index = tscore.index(max(tscore))
        tmp_iou_list = iou_with_anchors(
            np.array(tstart),
            np.array(tend), tstart[max_index], tend[max_index])
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                tmp_width = tend[max_index] - tstart[max_index]
                if tmp_iou > t1 + (t2 - t1) * tmp_width:
                    tscore[idx] = tscore[idx] * np.exp(-np.power(tmp_iou, iou_power) /
                                                       alpha)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        rindex.append(tindex[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tindex.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    newDf['rindex'] = rindex
    return newDf


def epic_video_post_process(cfg, video_list, result_dict, epoch, norm=False):
    """
    Post processing for part videos in epic dataset.
    Args:
        cfg (Config): the global config object.
        video_list (list): videos name list.
        result_dict (dict): results save to this dict
        epoch (int): epoch number of these proposals
    return:
        Processed action localization results for epic dataset.
    """
    select_score = cfg.LOCALIZATION.POST_PROCESS.SELECT_SCORE
    score_type = cfg.LOCALIZATION.POST_PROCESS.SCORE_TYPE
    clr_power = cfg.LOCALIZATION.POST_PROCESS.CLR_POWER
    reg_power = cfg.LOCALIZATION.POST_PROCESS.REG_POWER
    tca_power = cfg.LOCALIZATION.POST_PROCESS.TCA_POWER
    action_score_power = cfg.LOCALIZATION.POST_PROCESS.ACTION_SCORE_POWER
    action_key = 'label' if 'val' in cfg.TEST.TEST_SET else 'action'
    for video_name, duration in tqdm(video_list, total=len(video_list)):
        if cfg.OSS.CHECKPOINT_OUTPUT_PATH[:3] == "oss":
            video_path = os.path.join(cfg.LOCALIZATION.TEST_OUTPUT_DIR, "prop_results", video_name + ".pkl")
        else:
            video_path = os.path.join(cfg.OUTPUT_DIR, 
                                      _get_oss_path_prefix("prop_results", epoch, cfg.TEST.TEST_SET), 
                                      video_name + ".pkl")
        try:
            df = torch.load(video_path)
            heads = df[1][:-2]
            propos_data = [d[:-2] for d in df[0]]
            verb_noun = [d[-2:] for d in df[0]]
            df = pd.DataFrame(propos_data, columns=heads)
        except:
            logger.error("missed video proposals:{}".format(video_path))
            continue
        if score_type == 'cr':
            df['score'] = np.power(df.clr_score.values[:], clr_power) * np.power(df.reg_score.values[:], reg_power)
        elif score_type == 'se':
            df['score'] = df.xmin_score.values[:] * df.xmin_score.values[:]
        elif score_type == 'secr':
            df['score'] = df.clr_score.values[:] * df.reg_score.values[:] * df.xmin_score.values[:] * df.xmin_score.values[:]
        elif score_type == 'xwcr':
            df['score'] = np.power(df.clr_score.values[:], clr_power) * np.power(df.reg_score.values[:], reg_power) * np.power(df.tca_xw_score.values[:], tca_power)
        elif score_type == 'xwsecr':
            df['score'] = np.power(df.clr_score.values[:], clr_power) * np.power(df.reg_score.values[:], reg_power) * df.tca_xw_score.values[:] * df.tca_se_score.values[:]
        else:
            raise ValueError("unknown score_type: {}".format(score_type))
        df = df[df['score'] > select_score]
        if len(df) > 1:
            snms_alpha = cfg.LOCALIZATION.POST_PROCESS.SOFT_NMS_ALPHA
            snms_t1 = cfg.LOCALIZATION.POST_PROCESS.SOFT_NMS_LOW_THRES
            snms_t2 = cfg.LOCALIZATION.POST_PROCESS.SOFT_NMS_HIGH_THRES
            prop_num = int(duration / cfg.LOCALIZATION.POST_PROCESS.PROP_NUM_RATIO) + 1
            iou_power = cfg.LOCALIZATION.POST_PROCESS.IOU_POWER
            df = soft_nms(df, snms_alpha, snms_t1, snms_t2, prop_num, iou_power)

        df = df.sort_values(by="score", ascending=False)
        rindex = df.rindex.values[:]
        verb_noun = [verb_noun[idx] for idx in rindex]
        video_duration = duration
        proposal_list = []

        use_topk = 5
        for j in range(min(prop_num, len(df))):
            vn, vn_score = verb_noun[j]
            for k in range(use_topk):
                tmp_det = {}
                label_v, label_n = int(vn[k, 0]), int(vn[k, 1])
                tmp_det["score"] = df.score.values[j] * np.power(vn_score[k, 2], action_score_power)
                tmp_det[action_key] = "{},{}".format(label_v, label_n)
                tmp_det["verb"] = label_v
                tmp_det["noun"] = label_n
                tmp_det["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                      min(1, df.xmax.values[j]) * video_duration]
                proposal_list.append(tmp_det)
        result_dict[video_name] = proposal_list
