#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Epic-Kitchens features dataset. """

import os, sys
import random
import torch
import torch.utils.data
import utils.logging as logging
import json
import time
import math
import numpy as np
import oss2 as oss
import traceback
import torch.nn.functional as F
from datasets.base.epickitchen100 import Epickitchen100
from utils.bboxes_1d import (ioa_with_anchors, iou_with_anchors)
import utils.bucket as bu

from datasets.base.builder import DATASET_REGISTRY

logger = logging.get_logger(__name__)


def load_feature(path):
    """
    Load features from path or IO.
    Args:
        path (io.BytesIO or string): File path or io.
    """
    if type(path) is str:
        with open(path, 'rb') as f:
            data = torch.load(f)
    else:
        data = torch.load(path)
    return data


def str2sec(instr):
    """
    Transfer epic annotation strings to time.
    Args:
        instr (string): Time string read from annotation files.
    """
    data = instr.split(':')
    if len(data) == 2:
        h = 0
        m, s = data
    elif len(data) == 3:
        h, m, s = data
    return float(h) * 3600 + float(m)*60 + float(s)


@DATASET_REGISTRY.register()
class Epickitchen100localization(Epickitchen100):
    def __init__(self, cfg, split):
        """
        Init Localization dataset. 
        Args:
            cfg (Configs): global config object. details in utils/config.py
            split (string): train or val.
        """
        self.construct_set = cfg.TEST.TEST_SET if split not in 'training' else 'training'
        super(Epickitchen100localization, self).__init__(cfg, split)
        self.clip_sec = 5
        self.feature_stride_ratio = 2
        self.feature_stride = 0.13333333
        self.tscale = cfg.DATA.TEMPORAL_SCALE
        self.dscale = cfg.DATA.DURATION_SCALE if cfg.DATA.DURATION_SCALE > 0 else cfg.DATA.TEMPORAL_SCALE
        self.cls_files_root = cfg.DATA.CLASSIFIER_ROOT_DIR if hasattr(cfg.DATA, 'CLASSIFIER_ROOT_DIR') else ""
        self.load_cls_results = cfg.DATA.LOAD_CLASSIFIER_RES if hasattr(cfg.DATA, 'LOAD_CLASSIFIER_RES') else False
        self.boundary_offset = cfg.DATA.BOUNDARY_OFFSET if hasattr(cfg.DATA, 'BOUNDARY_OFFSET') else 0.0
        self.labels_type = cfg.DATA.LABELS_TYPE
        self._process_localization_dataset(cfg)
        self._init_temporal_tools()
        self._download_anno_json()
        self._construct_video_clips(cfg)

    def _download_anno_json(self):
        """
        Init annotation files and video-duration list for evaluation.
        """
        dataset_list_name = self.cfg.DATA.ANNO_NAME
        logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.construct_set))
        self.local_anno_file = os.path.join(self.cfg.OUTPUT_DIR, dataset_list_name)  + '{}'.format(self.cfg.LOCAL_RANK if hasattr(self.cfg, 'LOCAL_RANK') else 0)
        self.local_anno_file = self._get_object_to_file(os.path.join(self.anno_dir, dataset_list_name), self.local_anno_file, read_from_buffer=False)
        self._video_name_duration = []
        vid_name_list = []
        for s in self._samples:
            if s[0] not in vid_name_list:
                vid_name_list.append(s[0])
                self._video_name_duration.append([s[0], self._video_len_dict[s[0]]])

    def _filter_negtive_segments(self):
        """
        Filter too short action segments, which have no help for training.
        """
        _samples = []
        for s in self._samples:
            if str2sec(s[5]) - self.boundary_offset > str2sec(s[4]) + self.boundary_offset + 0.2:
                _samples.append(s)
        self._samples = _samples

    def _init_temporal_tools(self):
        """
        Init temporal tools for boundary matching map generation.
        Such as anchors and valid iou mask for boundary matching network.
        """
        match_map = []
        tscale = self.tscale
        dscale = self.dscale
        temporal_gap = 1.0 / tscale
        for idx in range(tscale):
            tmp_match_window = []
            xmin = temporal_gap * idx
            for jdx in range(1, dscale + 1):
                xmax = xmin + temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map  # duration is same in row, start is same in col
        self.anchor_xmin = [temporal_gap * (i-0.5) for i in range(tscale)]
        self.anchor_xmax = [temporal_gap * (i+0.5) for i in range(1, tscale + 1)]
        bm_mask = match_map.reshape(dscale, tscale, 2)[:, :, 1] <= 1.0
        self.iou_map_mask = torch.Tensor(bm_mask).float()

    def _load_video_length(self, cfg):
        """
        Load the true length of videos for sliding windows.
        Args:
            cfg (Configs): global config object. details in utils/config.py
        """
        self._video_len_dict = {}
        vid_length_file = cfg.DATA.VIDEO_LENGTH_FILE
        local_clips_file = os.path.join(cfg.OUTPUT_DIR, vid_length_file) + '{}'.format(cfg.LOCAL_RANK if hasattr(cfg, 'LOCAL_RANK') else 0)
        local_clips_file = self._get_object_to_file(os.path.join(cfg.DATA.ANNO_DIR, vid_length_file), local_clips_file, read_from_buffer=False)
        with open(local_clips_file, 'r') as f:
            for line in f:
                name, length = line.strip().split(',')
                self._video_len_dict[name.replace('.MP4', '')] = float(length)

    def _process_localization_dataset(self, cfg):
        """
        Generate sliding windows and matching their ground truth for training and validation.
        Args:
            cfg (Configs): global config object. details in utils/config.py
        """
        self._load_video_length(cfg)
        self._construct_video_anno()
        tal_stride = self.feature_stride_ratio * self.feature_stride
        segments_list = []
        for name, duration in self._video_len_dict.items():
            segment_len = self.tscale * tal_stride
            start_time = torch.arange(0, duration, self.dscale * tal_stride)
            if duration - start_time[-1] < 0.2:
                start_time = start_time[:-1]
            end_time = start_time + segment_len
            segs = torch.stack([start_time, end_time], dim=1).tolist()
            for s in segs:
                segments_list.append([name, s])
        _samples_list = []
        for name, seg in segments_list:
            if name in self._video_anno:
                if self.cfg.TRAIN.ENABLE:
                    gts = np.array(self._video_anno[name][0])
                    labels = np.array(self._video_anno[name][1])
                    ioa = ioa_with_anchors(gts[:, 0], gts[:, 1], seg[0], seg[1])
                    gts = gts - seg[0]
                    select_mask = ioa > 0.999
                    if select_mask.sum() > 0:
                        _samples_list.append([name, seg, gts[select_mask, :], labels[select_mask, :]])
                else:
                    _samples_list.append([name, seg, None, None])
        self._samples = _samples_list
        logger.info("Dataset {} split {} loaded. Length {} sliding windows.".format(self.dataset_name, self.construct_set, len(self._samples)))

    def _construct_video_anno(self):
        """
        Transfer string to time or label.
        """
        self._video_anno = {}
        for s in self._samples:
            if s[2] not in self._video_anno:
                self._video_anno[s[2]] = [[], []]
            self._video_anno[s[2]][0].append([str2sec(s[4]), str2sec(s[5])])
            if len(s) > 8:
                self._video_anno[s[2]][1].append([int(s[10]), int(s[12])])

    def _construct_video_clips(self, cfg):
        """
        We divide each videos to multi-clips, and each clip contains 5 seconds. 
        This function is used to generate video clips path for easy use.
        Args:
            cfg (Configs): global config object. details in utils/config.py
        """
        self._video_clips_dict = {}
        clips_list_file = cfg.DATA.CLIPS_LIST_FILE
        local_clips_file = os.path.join(cfg.OUTPUT_DIR, clips_list_file) + '{}'.format(cfg.LOCAL_RANK if hasattr(cfg, 'LOCAL_RANK') else 0)
        local_clips_file = self._get_object_to_file(os.path.join(cfg.DATA.ANNO_DIR, clips_list_file), local_clips_file, read_from_buffer=False)
        with open(local_clips_file, 'r') as f:
            for line in f:
                line = line.strip()
                video_name, start_time, end_time = line.split(',')
                if (float(end_time) - float(start_time)) / 1000.0 < 0.2:
                    continue
                full_name = '_'.join([video_name, start_time, end_time])
                self._video_clips_dict[(video_name, int(int(start_time)/1000))] = full_name + '.pkl'

    def _get_feature_files(self, seg_time, video_name, index):
        """
        We divide each videos to multi clips, and each clip contains 5s. 
        This function is used to get clips which contain seg_time.
        Args:
            seg_time (list): [start_time, end_time]
            video_name (str): video name.
            index (int): the index of the data in the self._samples list.
        Returns:
            features_list    (list): The path of features.
            class_res_list   (list): The path of classification results.
            features_time    (list): [start_time, end_time] for features
        """
        start_sec = math.floor(seg_time[0] / self.clip_sec) * self.clip_sec
        end_sec = math.ceil(seg_time[1] / self.clip_sec) * self.clip_sec
        features_list = []
        class_res_list = []
        for st in range(start_sec, end_sec, self.clip_sec):
            if (video_name, st) in self._video_clips_dict:
                vid_full_name = self._video_clips_dict[(video_name, st)]
                if type(self.data_root_dir) is list:
                    features_list.append([os.path.join(drd, vid_full_name) for drd in self.data_root_dir])
                else:
                    raise NotImplementedError("unknown self.data_root_dir:{}".format(self.data_root_dir))
                if self.load_cls_results and not self.cfg.TRAIN.ENABLE:
                    class_res_list.append(os.path.join(self.cls_files_root, vid_full_name))
        if len(features_list) == 0:
            print(seg_time, video_name, index)
        return features_list, class_res_list, [start_sec, float(vid_full_name.split('_')[-1][:-4])/1000.0]

    def _get_sample_info(self, index):
        """
        Get annotation and data path for training and validation.
        Args:
            index        (int): the index of the data in the self._samples list.
        Returns:
            sample_info  (dict):{
                'seg_time': (list),
                'gt_time': (np.array),
                'label': (np.array),
                'feature_time': (list),
                'video_name': (str),
                'feat_path' : (list),
                'cls_path' : (list),
            }
        """
        video_name, seg_time, gt_time, label = self._samples[index]
        feat_path, cls_path, feature_time = self._get_feature_files(seg_time, video_name, index)
        sample_info = {
            "seg_time": seg_time,
            "gt_time": gt_time,
            "label": label,
            "feature_time": feature_time,
            "video_name": video_name,
            "feat_path": feat_path,
            "cls_path": cls_path,
        }
        return sample_info

    def __getitem__(self, index):
        """
        Gets the specified data.
        Args:
            index (int): the index of the data in the self._samples list.
        Returns:
            meta_dict (dict): {
                "video": (tensor), 
                "verb_cls_data" : (tensor),
                "noun_cls_data" : (tensor),
                "video_name" : (str),
                "seg_time" : (tensor),
                "feature_len" : (int)
            }
            labels (dict): {
                "supervised": (dict)
            }
        """
        sample_info = self._get_sample_info(index)
        try:
            data_dict, file_to_remove, success = self._load_oss_data(
                sample_info
            )
        except Exception as e:
            success = False
            traceback.print_exc()
            logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                index, sample_info["feat_path"]
            ))
        if not success:
            return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
        bu.clear_tmp_file(file_to_remove)

        feature_data = data_dict['feature_data']
        verb_cls_data = data_dict['verb_cls_data']
        noun_cls_data = data_dict['noun_cls_data']
        feature_data, feature_len, feat_mask_in_global = self._transform_feature_scale(feature_data, sample_info['seg_time'], sample_info['feature_time'])
        if len(verb_cls_data) > 0:
            verb_cls_data, cls_len, _ = self._transform_feature_scale(verb_cls_data, sample_info['seg_time'], sample_info['feature_time'])
            noun_cls_data, cls_len, _ = self._transform_feature_scale(noun_cls_data, sample_info['seg_time'], sample_info['feature_time'])

        meta_dict = {'video': feature_data,
                     'verb_cls_data': verb_cls_data,
                     'noun_cls_data': noun_cls_data,
                     'video_name': sample_info['video_name'],
                     'seg_time': sample_info['seg_time'],
                     'feature_len': feature_len,
                     "index": index}
        lables = {}
        if self.cfg.DATA.LABELS_TYPE in ['bmn']:
            if self.cfg.TRAIN.ENABLE:
                start_map, end_map, iou_map, label_map = self._get_bmn_train_label(sample_info['gt_time'], sample_info['label'], sample_info['seg_time'])
                lables["supervised"] = {"start_map": start_map,
                                        "end_map": end_map, 
                                        "iou_map": iou_map, 
                                        "label_map": label_map,
                                        "mask": self.iou_map_mask}
            else:
                meta_dict['feat_mask_in_global'] = feat_mask_in_global
            meta_dict['mask'] = self.iou_map_mask
        return meta_dict, lables, index, {}

    def _load_oss_data(self, sample_info):
        """
        Load feature data and classification data.
        Args:
            sample_info (dict): feat_path for feature and cls_path for classification
        Returns:
            data_dict (dict): {
                "feature_data": (np.array),
                "verb_cls_data": (np.array),
                "noun_cls_data": (np.array),
            }
        """
        if self.cfg.TRAIN.ENABLE:
            num_retries = 1
        else:
            num_retries = 100
        feature_success = True
        feature_data_list = []
        feature_file_to_remove = []
        for i, path_list in enumerate(sample_info['feat_path']):
            video_data_list = []
            for path in path_list:
                _feature_data, _feature_file_to_remove, _feature_success = self._read_feature(
                        path, num_retries)
                feature_file_to_remove.extend(_feature_file_to_remove)
                feature_success = feature_success and _feature_success
                video_data_list.append(_feature_data)
            feature_data_list.append(np.concatenate(video_data_list, axis=1))
        cls_success = True
        verb_cls_data_list = []
        noun_cls_data_list = []
        cls_file_to_remove = []
        if len(sample_info['cls_path']) > 0:
            for i, path in enumerate(sample_info['cls_path']):
                _cls_data, _cls_file_to_remove, _cls_success = self._read_feature(
                        path, num_retries)
                cls_file_to_remove.extend(_cls_file_to_remove)
                cls_success = cls_success and _cls_success
                verb_cls_data_list.append(_cls_data['verb_class'])
                noun_cls_data_list.append(_cls_data['noun_class'])

        data_dict = {"feature_data": feature_data_list,
                     "verb_cls_data": verb_cls_data_list,
                     "noun_cls_data": noun_cls_data_list,
                     }
        return data_dict, feature_file_to_remove + cls_file_to_remove, feature_success and cls_success

    def _transform_feature_scale(self, feature, seg_time, feature_time):
        """
        We divide each videos to multi clips, and each clip contains 5s.
        For example, if the seg_time is [8.5, 16.1], which means that we need three clip features: [[5.0,10.0], [10.0, 15.0], [15.0, 20.0]]
        However,  we only need the features between [8.5, 16.1].
        We calculate the center point of the each feature, and only take the center time between [8.5, 16.1].
        Args:
            feature (list): features which contain seg_time.
            seg_time (list): [start_time, end_time] for sliding windows.
            feature_time (list): [start_time, end_time] for original feature.
        Returns:
            feature: (tensor)
            feature_len: (int)
            feat_mask_in_global: (tensor)
        """
        feature = np.concatenate(feature, axis=0)
        max_idx = seg_time[1]
        clip_interval = self.cfg.DATA.CLIP_INTERVAL
        clip_length = self._num_frames * self._sampling_rate

        center_idx = torch.arange(clip_interval-1, feature_time[1]*self.cfg.DATA.TARGET_FPS, clip_interval) / float(self.cfg.DATA.TARGET_FPS)
        feat_mask_in_global = (center_idx >= seg_time[0]) & (center_idx <= seg_time[1]) & (center_idx <= feature_time[1])
        select_center = (center_idx >= feature_time[0]) & (center_idx <= feature_time[1])
        center_idx = center_idx[select_center]
        select_center = (center_idx >= seg_time[0]) & (center_idx <= seg_time[1]) & (center_idx <= feature_time[1])
        if (feature.shape[0] - select_center.shape[0]) == 1:
            feature = feature[:-1]
        feature = torch.from_numpy(feature[select_center, :]).permute(1, 0)[None, :, :]
        tal_feature_len = round((seg_time[1]-seg_time[0]) / self.feature_stride)
        feature_len = feature.size(-1)
        if feature_len != tal_feature_len:
            empty_data = torch.zeros(1, feature.size(1), tal_feature_len)
            empty_data[0, :, :feature.size(2)] = feature[0, :, :tal_feature_len]
            feature = empty_data
        if feature_len == 0:
            feature = torch.zeros(1, self.cfg.DATA.NUM_INPUT_CHANNELS, 1)
        if self.tscale > 0:
            feature = torch.nn.functional.interpolate(feature, size=self.tscale, mode='linear', align_corners=True)[0]
            return feature, feature_len, feat_mask_in_global
        else:
            empty_data = torch.zeros(feature.size(1), self.tscale)
            empty_data[:, :feature.size(2)] = feature[0]
            return empty_data, feature_len, feat_mask_in_global

    def _read_feature(self, feature_path, num_retries):
        """
        Load feature from file.
        Args:
            feature_path (str): File path for a feature.
            num_retries (int): Retry time if download file failed.
        Returns:
            data: (tensor)
            file_to_remove: (list)
            success: (bool)
        """
        tmp_file_name = str(round(time.time() * 1000)) + feature_path.split('/')[-1]
        tmp_file = tmp_file_name
        read_retry = 10
        for tidx in range(read_retry):
            try:
                tmp_file = self._get_object_to_file(feature_path, tmp_file_name, read_from_buffer=True, num_retries=num_retries)
                _tries = 0
                while tmp_file is None:
                    print("trying download {} {} retries...".format(_tries+1, feature_path))
                    _tries += 1
                    tmp_file = self._get_object_to_file(feature_path, tmp_file_name, read_from_buffer=True, num_retries=num_retries)
                data = load_feature(tmp_file)
                success = True
                break
            except:
                traceback.print_exc()
                success = False
                data = None
                logger.info("Load file {} failed. {}/{} Retrying...".format(
                    feature_path, tidx+1, read_retry, 
                ))
        file_to_remove = [tmp_file] if feature_path[:3] == "oss" else [None]
        return data, file_to_remove, success

    def _norm_gt_boxes(self, gt_time, seg_time):
        """
        Norm ground truth time into [0,1].
        Args:
            gt_time (np.array): The shape is N*2. There have N ground truth action in seg_time.
            seg_time (list): [start_time, end_time] for sliding windows.
        Returns:
            gt_bbox: (np.array)
        """
        gt_bbox = []
        corrected_second = 1.0
        duration = seg_time[1] - seg_time[0]
        video_labels = gt_time
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info[0] / (duration * corrected_second)), 0)
            tmp_end = max(min(1, tmp_info[1] / (duration * corrected_second)), 0)
            gt_bbox.append([tmp_start, tmp_end])
        return np.array(gt_bbox)

    def _get_bmn_iou_map(self, gt_bbox, label):
        """
        Get the maps for boundary matching network.
        Args:
            gt_bbox (np.array): The shape is N*2. Normalized temporal bounding boxes.
            label (np.array): Action categories for N ground truth bounding boxes.
        Returns:
            gt_iou_map: (tensor)
            gt_label_map: (tensor)
        """
        gt_iou_map = []
        for j in range(len(gt_bbox)):
            tmp_start, tmp_end = gt_bbox[j, 0], gt_bbox[j, 1]
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.dscale, self.tscale])
            gt_iou_map.append(tmp_gt_iou_map)
        if len(gt_iou_map) == 0:
            gt_iou_map = np.zeros((1, self.dscale, self.tscale))
            labels_vector = np.zeros((2, self.dscale * self.tscale), dtype='long')
        else:
            gt_iou_map = np.array(gt_iou_map)
            labels_vector = label[np.argmax(gt_iou_map, axis=0).reshape(-1), :].transpose(1, 0)
        gt_label_map = torch.from_numpy(labels_vector.reshape([2, self.dscale, self.tscale]))
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)
        return gt_iou_map, gt_label_map

    def _get_start_end(self, gt_bbox):
        """
        Get the start and end sequences.
        Args:
            gt_bbox (np.array): The shape is N*2. Normalized temporal bounding boxes.
        Returns:
            match_score_start: (tensor)
            match_score_end: (tensor)
        """
        if len(gt_bbox) == 0:
            return np.zeros(self.tscale, dtype='float'), np.zeros(self.tscale, dtype='float')
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 / float(self.tscale)  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        match_score_start = []
        anchor_xmin = self.anchor_xmin
        anchor_xmax = self.anchor_xmax
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_start, match_score_end

    def _get_bmn_train_label(self, gt_time, label, seg_time):
        """
        Get full bmn training labels.
        Args:
            gt_time (np.array): The shape is N*2. The ground truth temporal bounding boxes.
            label (np.array): Action categories for N ground truth bounding boxes.
            seg_time (list): [start_time, end_time] for sliding windows.
        Returns:
            match_score_start: (tensor)
            match_score_end: (tensor)
            gt_iou_map: (tensor)
            gt_label_map: (tensor)
        """
        gt_bbox = self._norm_gt_boxes(gt_time, seg_time)
        gt_iou_map, gt_label_map = self._get_bmn_iou_map(gt_bbox, label)
        match_score_start, match_score_end = self._get_start_end(gt_bbox)
        return match_score_start, match_score_end, gt_iou_map, gt_label_map
