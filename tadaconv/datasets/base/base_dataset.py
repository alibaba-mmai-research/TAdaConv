#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" BaseVideoDataset object to be extended for specific datasets. """

import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.utils.dlpack as dlpack
import tadaconv.utils.logging as logging

import re
import abc
import time
import random
import decord
import traceback
import numpy as np
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge('native')

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from tadaconv.datasets.utils.transformations import (
    ColorJitter, 
    KineticsResizedCrop
)
from tadaconv.datasets.utils.random_erasing import RandomErasing

from tadaconv.sslgenerators.builder import build_ssl_generator

import tadaconv.utils.bucket as bu

logger = logging.get_logger(__name__)

class BaseVideoDataset(torch.utils.data.Dataset):
    """
    The BaseVideoDataset object provides a base object for all the video/image/video-text datasets.
    Abstract methods are provided for completion in the specific datasets.
    Necessary methods for all datasets such as "_decode_video", "_decode_image", 
    "__getitem__" (with standard procedure for loading the data) as well as sampling methods 
    such as "_interval_based_sampling" and "_segment_based_sampling" are implemented. 
    The specific video datasets can be extended from this dataset according to different needs.
    """
    def __init__(self, cfg, split):
        """
        For initialization of the dataset, the global cfg and the split need to provided.
        Args:
            cfg     (Config): The global config object.
            split   (str): The split, e.g., "train", "val", "test"
        """
        self.cfg            = cfg
        self.split          = split
        self.data_root_dir  = cfg.DATA.DATA_ROOT_DIR
        self.anno_dir       = cfg.DATA.ANNO_DIR

        if self.split in ["train", "val"]:
            self.dataset_name = cfg.TRAIN.DATASET
            self._num_clips = 1
        elif self.split in ["test", "submission"]:
            self.dataset_name = cfg.TEST.DATASET
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        else:
            raise NotImplementedError("Split not supported")

        self._num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self._sampling_rate = cfg.DATA.SAMPLING_RATE

        self.gpu_transform = cfg.AUGMENTATION.USE_GPU       # whether or not to perform the transform on GPU

        self.decode = self._decode_video                    # decode function, decode videos by default

        self.buckets = {}

        # if set to true, _pre_transformation_config will be called before every transformations
        # this is used in the testset, where cropping positions are set for the controlled crop
        self._pre_transformation_config_required = False    
        self._construct_dataset(cfg)
        self._config_transform()

        # configures the pre-training
        if self.cfg.PRETRAIN.ENABLE:
            self.ssl_generator = build_ssl_generator(self.cfg, split)
            # NUM_CLIPS_PER_VIDEO specifies the number of clips decoded for each video
            # for contrastive learning, NUM_CLIPS_PER_VIDEO=2
            # for other ssl, NUM_CLIPS_PER_VIDEO=1
            self.num_clips_per_video = cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO

    @abc.abstractmethod
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                Things that must be included are:
                "path" indicating the target's path w.r.t. index
                "supervised_label" indicating the class of the target 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_ssl_label(self, frames):
        """
        Uses cfg to obtain ssl label.
        Returns:
            ssl_label (dict): self-supervised labels
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        raise NotImplementedError

    def _get_video_frames_list(self, vid_length, vid_fps, clip_idx, random_sample=True):
        """
        Returns the list of frame indexes in the video for decoding. 
        Args:
            vid_length (int): video length
            clip_idx (int): clip index, -1 if random sampling (interval based sampling)
            num_clips (int): overall number of clips for clip_idx != -1 (interval based sampling) 
            num_frames (int): number of frames to sample 
            interval (int): the step size for interval based sampling (interval based sampling)
            random_sample (int): whether to randomly sample one frame from each segment (segment based sampling)
        Returns:
            frame_id_list (list): indicates which frames to sample from the video
        """
        if self.cfg.PRETRAIN.ENABLE and self.split == "train":
            return self._custom_sampling(vid_length, vid_fps, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate, random_sample)
        else:
            if self.cfg.DATA.SAMPLING_MODE == "interval_based":
                return self._interval_based_sampling(vid_length, vid_fps, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate)
            elif self.cfg.DATA.SAMPLING_MODE == "segment_based":
                return self._segment_based_sampling(vid_length, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, random_sample)
            else:
                raise NotImplementedError

    def _construct_dataset(self, cfg):
        """
        Constructs the dataset according to the global config object.
        Currently supports reading from csv, json and txt.
        Args:
            cfg (Config): The global config object.
        """
        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()

        try:
            logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.split))
            local_file = os.path.join(cfg.OUTPUT_DIR, dataset_list_name)
            local_file = self._get_object_to_file(os.path.join(self.anno_dir, dataset_list_name), local_file)
            if local_file[-4:] == ".csv":
                import pandas
                lines = pandas.read_csv(local_file)
                for line in lines.values.tolist():
                    for idx in range(self._num_clips):
                        self._samples.append(line)
                        self._spatial_temporal_index.append(idx)
            elif local_file[-4:] == "json":
                import json
                with open(local_file, "r") as f:
                    lines = json.load(f)
                for line in lines:
                    for idx in range(self._num_clips):
                        self._samples.append(line)
                        self._spatial_temporal_index.append(idx)
            else:
                with open(local_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        for idx in range(self._num_clips):
                            self._samples.append(line.strip())
                            self._spatial_temporal_index.append(idx)
            logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
        except:
            raise ValueError("Data list {} not found.".format(os.path.join(self.anno_dir, dataset_list_name)))
        
        # validity check    
        assert len(self._samples) != 0, "Empty sample list {}".format(os.path.join(self.anno_dir, dataset_list_name))

    def _read_video(self, video_path, index):
        """
        Wrapper for downloading the video and generating the VideoReader object for reading the video. 
        Args: 
            video_path (str): video path to read the video from. Can in OSS form or in local hard drives.
            index      (int):  for debug.
        Returns:
            vr              (VideoReader):  VideoReader object wrapping the video.
            file_to_remove  (list):         list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool):         flag for the indication of success or not.
        """
        tmp_file = str(round(time.time() * 1000)) + video_path.split('/')[-1]  
        try:
            vr = None
            tmp_file = self._get_object_to_file(video_path, tmp_file, read_from_buffer=True, num_retries=1 if self.split == "train" else 20)
            vr = VideoReader(tmp_file)
            success = True
        except:
            success = False
        file_to_remove = [tmp_file] if video_path[:3] == "oss" else [None] # if not downloaded from oss, then no files need to be removed
        return vr, file_to_remove, success

    def _decode_video(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the video given the sample info.
        Args: 
            sample_info         (dict): containing the "path" key specifying the location of the video.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
        Returns:
            data            (dict): key "video" for the video data.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        vr, file_to_remove, success =  self._read_video(path, index)

        if not success:
            return vr, file_to_remove, success

        if self.split == "train":
            clip_idx = -1
            self.spatial_idx = -1
        elif self.split == "val":
            clip_idx = -1
            self.spatial_idx = 0
        elif self.split == "test" or self.split == "submission":
            clip_idx = self._spatial_temporal_index[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                self.spatial_idx = 0
            else:
                self.spatial_idx = self._spatial_temporal_index[index] % self.cfg.TEST.NUM_SPATIAL_CROPS

        frame_list= []
        for idx in range(num_clips_per_video):
            # for each clip in the video, 
            # a list is generated before decoding the specified frames from the video
            list_ = self._get_video_frames_list(
                len(vr),
                vr.get_avg_fps(),
                clip_idx,
                random_sample=True if self.split=="train" else False 
            )
            frames = None
            frames = dlpack.from_dlpack(vr.get_batch(list_).to_dlpack()).clone()
            frame_list.append(frames)
        frames = torch.stack(frame_list)
        if num_clips_per_video == 1:
            frames = frames.squeeze(0)
        del vr
        return {"video": frames}, file_to_remove, True

    def _read_image(self, path, index):
        """
        Wrapper for downloading the image and generating the PIL.Image object for reading the image. 
        Args: 
            path    (str): image path to read the image from. Can in OSS form or in local hard drives.
            index   (int):  for debug.
        Returns:
            img             (PIL.Image):    image object for further processing.
            file_to_remove  (list):         list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool):         flag for the indication of success or not.
        """
        tmp_file = str(round(time.time() * 1000)) + path.split('/')[-1]  
        for tmp in range(10):
            try:
                img = None
                tmp_file = self._get_object_to_file(path, tmp_file, read_from_buffer=True)
                if isinstance(tmp_file, str):
                    with open(tmp_file, 'rb') as f:
                        img = Image.open(f).convert('RGB')
                else:
                    img = Image.open(tmp_file).convert('RGB')
                success = True
                break
            except:
                success = False
        file_to_remove = [tmp_file] if path[:3] == "oss" else [None]
        return img, file_to_remove, success

    def _decode_image(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the image given the sample info.
        Args: 
            sample_info         (dict): containing the "path" key specifying the location of the image.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
                                        specifically in this function, num_clips_per_video does not matter since all things to be decoded is one image.
        Returns:
            data            (dict): key "video" for the image data.
                                    because this is a video database, the images will be in the shape of 1,H,W,C before further processing.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        img, tmp_file, success = self._read_image(path, index)

        if not success:
            return None, tmp_file, success

        frame = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0], len(img.getbands()))
        frame = frame.unsqueeze(0) # 1, H, W, C
        return {"video":frame}, tmp_file, True

    def __getitem__(self, index):
        """
        Gets the specified data.
        Args:
            index (int): the index of the data in the self._samples list.
        Returns:
            frames (dict): {
                "video": (tensor), 
                "text_embedding" (optional): (tensor)
            }
            labels (dict): {
                "supervised": (tensor),
                "self-supervised" (optional): (...)
            }
        """
        sample_info = self._get_sample_info(index)

        # decode the data
        retries = 1 if self.split == "train" else 10
        for retry in range(retries):
            try:
                data, file_to_remove, success = self.decode(
                    sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                )
                break
            except Exception as e:
                success = False
                traceback.print_exc()
                logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                    retry+1, retries, index, sample_info["path"]
                ))

        if not success:
            return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)

        if self.gpu_transform:
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
        if self._pre_transformation_config_required:
            self._pre_transformation_config()
        
        labels = {}
        labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
        if self.cfg.PRETRAIN.ENABLE:
            # generates the different augmented samples for pre-training
            try:
                data, labels["self-supervised"] = self.ssl_generator(data, index)
            except Exception as e:
                traceback.print_exc()
                print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                    index, sample_info["path"], data["video"].shape
                ))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
        else:
            # augment the samples for supervised training
            labels["self-supervised"] = {}
            if "flow" in data.keys() and "video" in data.keys():
                data = self.transform(data)
            elif "video" in data.keys():
                data["video"] = self.transform(data["video"]) # C, T, H, W = 3, 16, 240, 320, RGB
        
        if (self.split == "train" and \
            not self.cfg.PRETRAIN.ENABLE and \
            "ssv2" in self.cfg.TRAIN.DATASET and \
            self.cfg.AUGMENTATION.SSV2_FLIP):
            if random.random() < 0.5:
                data["video"] = torchvision.transforms._functional_video.hflip(data["video"])
                label_transforms = {
                    86: 87,
                    87: 86,
                    93: 94,
                    94: 93,
                    166: 167,
                    167: 166
                }
                if labels["supervised"] in label_transforms.keys():
                    labels["supervised"] = label_transforms[labels["supervised"]]
        
        # if the model is SlowFast, generate two sets of inputs with different framerates.
        if self.cfg.VIDEO.BACKBONE.META_ARCH == "Slowfast":
            slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1]//self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA+1).long()[:-1]
            fast_frames = data["video"].clone()
            slow_frames = data["video"][:,slow_idx,:,:].clone()
            data["video"] = [slow_frames, fast_frames]
        bu.clear_tmp_file(file_to_remove)

        return data, labels, index, {}
    
    def _get_object_to_file(self, obj_file: str, local_file, read_from_buffer=False, num_retries=10):
        """
        Wrapper for downloading the file object.
        Args:
            obj_file         (str):  the target file to be downloaded (if it starts by "oss").
            local_file       (str):  the local file to store the downloaded file.
            read_from_butter (bool): whether or not to directly download to the memory
            num_retries      (int):  number of retries.
        Returns:
            str or BytesIO depending on the read_from_buffer flag
            if read_from_buffer==True:
                returns BytesIO 
            else:
                returns str (indicating the location of the specified file)
        """
        if obj_file[:3] == "oss":
            bucket_name = obj_file.split('/')[2]
            if bucket_name not in self.buckets.keys():
                self.buckets[bucket_name] = self._initialize_oss(bucket_name)
            if read_from_buffer:
                local_file = bu.read_from_buffer(
                    self.buckets[bucket_name],
                    obj_file,
                    bucket_name,
                    num_retries
                )
            else:
                bu.read_from_bucket(
                    self.buckets[bucket_name],
                    obj_file,
                    local_file,
                    bucket_name,
                    num_retries
                )
            return local_file
        else:
            return obj_file
    
    def _initialize_oss(self, bucket_name):
        """
        Initializes the oss bucket.
        Currently supporting two OSS accounts.
        """
        if hasattr(self.cfg.OSS, "SECONDARY_DATA_OSS") and\
            self.cfg.OSS.SECONDARY_DATA_OSS.ENABLE and\
            bucket_name in self.cfg.OSS.SECONDARY_DATA_OSS.BUCKETS:
            return bu.initialize_bucket(
                self.cfg.OSS.SECONDARY_DATA_OSS.KEY, 
                self.cfg.OSS.SECONDARY_DATA_OSS.SECRET,
                self.cfg.OSS.SECONDARY_DATA_OSS.ENDPOINT,
                bucket_name
            )
        else:
            return bu.initialize_bucket(
                self.cfg.OSS.KEY, 
                self.cfg.OSS.SECRET,
                self.cfg.OSS.ENDPOINT,
                bucket_name
            )

    def __len__(self):
        """
        Returns the number of samples.
        """
        if hasattr(self.cfg.TRAIN, "NUM_SAMPLES") and self.split == 'train':
            return self.cfg.TRAIN.NUM_SAMPLES
        else:
            return len(self._samples)

    # -------------------------------------- Sampling Utils --------------------------------------
    def _interval_based_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval):
        """
        Generates the frame index list using interval based sampling.
        Args:
            vid_length  (int): the length of the whole video (valid selection range).
            vid_fps     (int): the original video fps
            clip_idx    (int): -1 for random temporal sampling, and positive values for sampling specific clip from the video
            num_clips   (int): the total clips to be sampled from each video. 
                                combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
            num_frames  (int): number of frames in each sampled clips.
            interval    (int): the interval to sample each frame.
        Returns:
            index (tensor): the sampled frame indexes
        """
        if num_frames == 1:
            index = [random.randint(0, vid_length-1)]
        else:
            # transform FPS
            clip_length = num_frames * interval * vid_fps / self.cfg.DATA.TARGET_FPS

            max_idx = max(vid_length - clip_length, 0)
            if clip_idx == -1: # random sampling
                start_idx = random.uniform(0, max_idx)
            else:
                if num_clips == 1:
                    start_idx = max_idx / 2
                else:
                    start_idx = max_idx * clip_idx / num_clips
            if self.cfg.DATA.MINUS_INTERVAL:
                end_idx = start_idx + clip_length - interval
            else:
                end_idx = start_idx + clip_length - 1

            index = torch.linspace(start_idx, end_idx, num_frames)
            index = torch.clamp(index, 0, vid_length-1).long()

        return index
    
    def _segment_based_sampling(self, vid_length, clip_idx, num_clips, num_frames, random_sample):
        """
        Generates the frame index list using segment based sampling.
        Args:
            vid_length    (int):  the length of the whole video (valid selection range).
            clip_idx      (int):  -1 for random temporal sampling, and positive values for sampling specific clip from the video
            num_clips     (int):  the total clips to be sampled from each video. 
                                    combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
            num_frames    (int):  number of frames in each sampled clips.
            random_sample (bool): whether or not to randomly sample from each segment. True for train and False for test.
        Returns:
            index (tensor): the sampled frame indexes
        """
        index = torch.zeros(num_frames)
        index_range = torch.linspace(0, vid_length, num_frames+1)
        for idx in range(num_frames):
            if random_sample:
                index[idx] = random.uniform(index_range[idx], index_range[idx+1])
            else:
                if num_clips == 1:
                    index[idx] = (index_range[idx] + index_range[idx+1]) / 2
                else:
                    index[idx] = index_range[idx] + (index_range[idx+1] - index_range[idx]) * (clip_idx+1) / num_clips
        index = torch.round(torch.clamp(index, 0, vid_length-1)).long()

        return index

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
                transforms.RandomHorizontalFlipVideo()
            ]
            
            if self.cfg.DATA.TRAIN_JITTER_SCALES[0] <= 1:
                std_transform_list += [transforms.RandomResizedCropVideo(
                        size=self.cfg.DATA.TRAIN_CROP_SIZE,
                        scale=[
                            self.cfg.DATA.TRAIN_JITTER_SCALES[0],
                            self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                        ],
                        ratio=self.cfg.AUGMENTATION.RATIO
                    ),]
            else:
                std_transform_list += [KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]
            if self.cfg.AUGMENTATION.AUTOAUGMENT.ENABLE:
                from tadaconv.datasets.utils.auto_augment import creat_auto_augmentation
                std_transform_list.append(creat_auto_augmentation(self.cfg.AUGMENTATION.AUTOAUGMENT.TYPE, self.cfg.DATA.TRAIN_CROP_SIZE, self.cfg.DATA.MEAN))
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        color=self.cfg.AUGMENTATION.COLOR_P,
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
