#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Something-Something-V2 dataset. """

import os
import random
import torch
import torch.utils.data
import tadaconv.utils.logging as logging

import time
import oss2 as oss

import tadaconv.utils.bucket as bu
from tadaconv.datasets.base.builder import DATASET_REGISTRY
from tadaconv.datasets.base.base_dataset import BaseVideoDataset

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Ssv2(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Ssv2, self).__init__(cfg, split) 
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
        
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        name = "something-something-v2-{}-with-label.json".format(
            "train" if self.split == "train" else "validation",
        )
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                "path": indicating the target's path w.r.t. index
                "supervised_label": indicating the class of the target 
        """
        class_ = self._samples[index]["label_idx"]
        video_path = os.path.join(self.data_root_dir, self._samples[index]["id"]+".mp4")
        sample_info = {
            "path": video_path,
            "supervised_label": class_,
        }
        return sample_info

    def _pre_transformation_config(self):
        """
        Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)
    
    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        pass # making python happy

    def _get_ssl_label(self):
        pass # making python happy