#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" ImageNet dataset. """

import os
import random
import torch
import torch.utils.data
import utils.logging as logging

import time
import oss2 as oss

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from datasets.utils.transformations import (
    ColorJitter, 
    AutoResizedCropVideo
)

from datasets.base.base_dataset import BaseVideoDataset
from datasets.utils.random_erasing import RandomErasing

import utils.bucket as bu

from datasets.base.builder import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Imagenet(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Imagenet, self).__init__(cfg, split) 
        self.decode = self._decode_image
        
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        if self.cfg.PRETRAIN.ENABLE:
            if self.split == "train":
                name = "imagenet_train_S{}.txt".format(
                    self.cfg.PRETRAIN.IMAGENET_DATA_SIZE, 
                )
            else:
                name = "imagenet_val.txt"        
        else:
            name = "imagenet_{}.txt".format(
                "train" if self.split == "train" else "val"
            )
        logger.info("Reading image list from file: {}".format(name))
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
        img_path, class_, = self._samples[index].strip().split(" ")
        class_ = int(class_)
        img_path = os.path.join(self.data_root_dir, "imagenet_{}".format(
            "train" if self.split == "train" else "val"
        ),  img_path)
        sample_info = {
            "path": img_path,
            "supervised_label": class_,
        }
        return sample_info
    
    def _config_transform(self):
        """
        Configs the transform for the dataset.
        For train, we apply random cropping, random horizontal flip, random color jitter (optionally),
            normalization and random erasing (optionally).
        For val and test, we apply controlled spatial cropping and normalization.
        The transformations are stored as a callable function to "self.transforms".

        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomResizedCropVideo(
                    size=self.cfg.DATA.TRAIN_CROP_SIZE
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
        elif self.split == 'val' or self.split == 'test':
            self.resize_video = AutoResizedCropVideo(
                    size=self.cfg.DATA.TEST_CROP_SIZE,
                    scale=[
                        0.875, 0.875  
                    ],
                    mode="cc",
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
