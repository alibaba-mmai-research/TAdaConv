#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Epic-Kitchens dataset. """

import os
import random
import torch
import torch.utils.data
import utils.logging as logging

import time
import oss2 as oss

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch.nn.functional as F
from datasets.utils.transformations import (
    ColorJitter, 
    KineticsResizedCrop
)

from datasets.base.base_dataset import BaseVideoDataset
from datasets.utils.random_erasing import RandomErasing

import utils.bucket as bu

from datasets.base.builder import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Epickitchen100(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Epickitchen100, self).__init__(cfg, split) 
        if (self.split == "test" or self.split == "submission") and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
        
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            dataset_list_name (str)
        """
        if self.split == "train":
            if self.cfg.TRAIN.TRAIN_VAL_COMBINE:
                train_list = "train_val"
            else:
                train_list = "train"
        name = "EPIC_100_{}.csv".format(
            train_list if self.split == "train" else "validation" if not self.split == "submission" else "test_timestamps",
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
                "name": the name of the video
                "path": the path of the video for the specified index
                "verb_class": verb label of the video
                "noun_class": noun label of the video
        """
        if not self.split == "submission":
            video_name  = self._samples[index][0]
            verb_class  = self._samples[index][10]
            noun_class  = self._samples[index][12]
            video_path  = os.path.join(self.data_root_dir, video_name+".MP4")
        else:
            # if the split is submission, then no label is available
            # we simply set the verb class and the noun class to zero
            video_name  = self._samples[index][0]
            verb_class  = 0
            noun_class  = 0
            video_path  = os.path.join(self.data_root_dir, video_name+".MP4")
        
        if self.cfg.DATA.MULTI_LABEL or not hasattr(self.cfg.DATA, "TRAIN_VERSION"):
            supervised_label = {
                "verb_class": verb_class,
                "noun_class": noun_class
            }
        else:
            if self.cfg.DATA.TRAIN_VERSION == "only_train_verb":
                supervised_label = verb_class
            elif self.cfg.DATA.TRAIN_VERSION == "only_train_noun":
                supervised_label = noun_class

        sample_info = {
            "name": video_name,
            "path": video_path,
            "supervised_label": supervised_label
        }
        return sample_info
    
    def _config_transform(self):
        """
        Configs the transform for the dataset.
        For train, we apply random cropping, random horizontal flip, random color jitter (optionally),
            normalization and random erasing (optionally).
        For val and test, we apply controlled spatial cropping and normalization.
        The transformations are stored as a callable function to "self.transforms".
        """
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list = [
                transforms.ToTensorVideo(),
                KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),
                transforms.RandomHorizontalFlipVideo()
            ]
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        ),
                )
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
            ]
            self.transform = Compose(std_transform_list)
        elif self.split == 'val' or self.split == 'test' or self.split == "submission":
            self.resize_video = KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)


    def _pre_transformation_config(self):
        """
        Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)
    
