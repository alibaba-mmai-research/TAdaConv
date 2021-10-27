#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Generator Class for Self-supervised Motion Representation Learning. """

import os
import torch
import random
import utils.logging as logging
import torchvision.transforms._functional_video as F

from sslgenerators.builder import SSL_GENERATOR_REGISTRY

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from datasets.utils.transformations import ColorJitter

logger = logging.get_logger(__name__)

@SSL_GENERATOR_REGISTRY.register()
class MoSIGenerator(object):
    """
    Generator for pseudo camera motions with static masks in MoSI.

    See paper "Self-supervised Motion Learning from Static Images",
    Huang et al. 2021 (https://arxiv.org/abs/2104.00240) for details.

    The MoSI generator process goes as follows:
    (a) In the initialization stage, a `speed_set` is generated according
    to the config. 
    (b) In the training stage, each speed in the `speed_set` is used to 
    generate a sample from the given data. 
    """
    def __init__(self, cfg, split):
        """
        Args: 
            cfg (Config): global config object. 
            split (str): the data split, e.g., "train", "val", "test"
        """
        self.cfg = cfg
        self.crop_size = cfg.DATA.TRAIN_CROP_SIZE
        self.num_speeds = cfg.VIDEO.HEAD.NUM_CLASSES
        self.distance_jitter = cfg.PRETRAIN.DISTANCE_JITTER
        self.label_mode = cfg.PRETRAIN.LABEL_MODE
        self.num_frames  = cfg.PRETRAIN.NUM_FRAMES
        self.split = split
        self.static_mask_enable = cfg.PRETRAIN.STATIC_MASK
        self.aspect_ratio = cfg.PRETRAIN.ASPECT_RATIO
        self.mask_size_ratio = cfg.PRETRAIN.MASK_SIZE_RATIO

        if type(self.crop_size) is list:
            assert len(self.crop_size) <= 2
            if len(self.crop_size) == 2:
                assert self.crop_size[0] == self.crop_size[1]
                self.crop_size = self.crop_size[0]
    
        assert (len(self.distance_jitter) == 2) and (
            self.distance_jitter[0] <= self.distance_jitter[1]
        )

        self.initialize_speed_set()
        self.labels = self.label_generator()
        self.config_transform()
        
    def initialize_speed_set(self):
        """
        Initialize speed set for x and y separately. 
        Initialized speed set is a list of lists [speed_x, speed_y].

        First a set of all speeds are generated as `speed_all`, then 
        the required speeds are taken from the `speed_all` according 
        to the config on MoSI to generate the `speed_set` for MoSI.
        """

        # Generate all possible speed combinations
        self.speed_all = []
        _zero_included = False

        # for example, when the number of classes is defined as 5,
        # then the speed range for each axis is [-2, -1, 0, 1, 2]
        # negative numbers indicate movement in the negative direction
        self.speed_range = (
            torch.linspace(0, self.num_speeds-1, self.num_speeds) - self.num_speeds // 2
        ).long()
        self.speed_min = int(min(self.speed_range))

        for x in self.speed_range:
            for y in self.speed_range:
                x,y = [int(x), int(y)]
                if x==0 and y==0:
                    if _zero_included:
                        continue
                    else:
                        _zero_included = True
                if self.cfg.PRETRAIN.DECOUPLE and x*y != 0: 
                    # if decouple, then one of x,y must be 0
                    continue
                self.speed_all.append(
                    [x, y]
                )
        
        # select speeds from all speed combinations
        self.speed_set = []
        assert self.cfg.PRETRAIN.DATA_MODE is not None
        if self.cfg.PRETRAIN.DECOUPLE:
            """
            Decouple means the movement is only on one axies. Therefore, at least one of
            the speed is zero.
            """
            if "x" in self.cfg.PRETRAIN.DATA_MODE:
                for i in range(len(self.speed_all)):
                    if self.speed_all[i][0] != 0: # speed of x is not 0
                        self.speed_set.append(self.speed_all[i])
            if "y" in self.cfg.PRETRAIN.DATA_MODE:
                for i in range(len(self.speed_all)):
                    if self.speed_all[i][1] != 0: # speed of y is not 0
                        self.speed_set.append(self.speed_all[i])
        else:
            # if not decoupled, all the speeds in the speed set is added in the speed set
            if "x" in self.cfg.PRETRAIN.DATA_MODE and "y" in self.cfg.PRETRAIN.DATA_MODE:
                self.speed_set = self.speed_all
            else:
                raise NotImplementedError("Not supported for data mode {} when DECOUPLE is set to true.".format(self.cfg.PRETRAIN.DATA_MODE))
            
        if self.cfg.PRETRAIN.DECOUPLE and not self.cfg.PRETRAIN.ZERO_OUT:
            # add speed=0
            self.speed_set.append([0,0])     

    def sample_generator(self, data, index):
        """
        Generate different MoSI samples for the data.
        Args:
            data (dict): the dictionary that contains a "video" key for the 
                decoded video data.
            index (int): the index of the video data.
        """
        out = []
        frames = data["video"]
        for speed_idx, speed in enumerate(self.speed_set):
            # generate all the samples according to the speed set
            num_input_frames, h, w, c = frames.shape
            frame_idx = random.randint(0, num_input_frames-1)
            selected_frame = frames[frame_idx] # H, W, C

            # standardize the frame size
            if self.cfg.PRETRAIN.FRAME_SIZE_STANDARDIZE_ENABLE: 
                selected_frame = self.frame_size_standardize(selected_frame)
            
            # generate the sample index 
            h, w, c = selected_frame.shape
            speed_x, speed_y = speed
            start_x, end_x = self.get_crop_params(speed_x/(self.num_speeds//2), w)
            start_y, end_y = self.get_crop_params(speed_y/(self.num_speeds//2), h)
            intermediate_x = (torch.linspace(start_x, end_x, self.num_frames).long()).clamp_(0, w-self.crop_size)
            intermediate_y = (torch.linspace(start_y, end_y, self.num_frames).long()).clamp_(0, h-self.crop_size)
            
            frames_out = torch.empty(
                self.num_frames, self.crop_size, self.crop_size, c, device=frames.device, dtype=frames.dtype
            )

            for t in range(self.num_frames):
                frames_out[t] = selected_frame[
                    intermediate_y[t]:intermediate_y[t]+self.crop_size, intermediate_x[t]:intermediate_x[t]+self.crop_size, :
                ]

            # performs augmentation on the generated image sequence
            if self.transform is not None:
                frames_out = self.transform(frames_out)
            
            # applies static mask
            if self.static_mask_enable:
                frames_out = self.static_mask(frames_out)
            out.append(frames_out)
        out = torch.stack(out)
        data["video"] = out
        return data
            
    def label_generator(self):
        """
        Generates the label for the MoSI.
        `separate` label is used for separate prediction on the two axes,
            i.e., two classification heads for each axis.
        'joint' label is used for joint prediction on the two axes.
        """
        if self.label_mode == 'separate':
            return self.generate_separate_labels()
        elif self.label_mode == 'joint':
            return self.generate_joint_labels()
            
    def generate_separate_labels(self):
        """
        Generates labels for separate prediction.
        """
        label_x = []
        label_y = []
        for speed_idx, speed in enumerate(self.speed_set):
            speed_x, speed_y = speed
            speed_x_label = speed_x - self.speed_min - (speed_x>0)*(self.cfg.PRETRAIN.ZERO_OUT)
            speed_y_label = speed_y - self.speed_min - (speed_y>0)*(self.cfg.PRETRAIN.ZERO_OUT)
            label_x.append(speed_x_label)
            label_y.append(speed_y_label)
        logger.info("LABELS:\nx: {},\ny: {}".format(label_x, label_y))
        return {
            "move_x": torch.tensor(label_x),
            "move_y": torch.tensor(label_y)
        }

    def generate_joint_labels(self):
        """
        Generates labels for joint prediction.
        """
        correspondence = "SPEED CORRESPONDENCE:\n"
        self.correspondence = []
        for speed_idx, speed in enumerate(self.speed_set):
            correspondence += "{}: {}\n".format(speed_idx, speed)
            self.correspondence.append(speed)
        logger.info(correspondence)
        return {
            "move_joint": torch.linspace(
                0, len(self.speed_set)-1, len(self.speed_set), dtype=torch.int64
            )
        }
            
    def get_crop_params(self, speed_factor, total_length):
        """
        Returns crop parameters.
        Args:
            speed_factor (float): frac{distance_to_go}{total_distance}
            total_length (int): length of the side
        """
        if speed_factor == 0:
            total_length>=self.crop_size, ValueError("Total length ({}) should not be less than crop size ({}) for speed {}.".format(
                total_length, self.crop_size, speed_factor
            ))
        else:
            assert total_length>self.crop_size, ValueError("Total length ({}) should be larger than crop size ({}) for speed {}.".format(
                total_length, self.crop_size, speed_factor
            ))
        assert abs(speed_factor) <= 1, ValueError("Speed factor should be smaller than 1. But {} was given.".format(speed_factor))

        distance_factor = self.get_distance_factor(speed_factor) if self.split == 'train' else 1
        distance = (total_length - self.crop_size) * speed_factor * distance_factor
        start_min = max(
            0, 0-distance 
        ) # if distance > 0, move right or down, start_x_min=0
        start_max = min(
            (total_length-self.crop_size),
            (total_length-self.crop_size)-distance
        ) # if distance > 0, move right or down, start_x_max = (w-crop_size)-distance
        start = random.randint(int(start_min), int(start_max)) if self.split == 'train' else (total_length-self.crop_size-distance)//2
        end   = start + distance
        return start, end

    def get_distance_factor(self, speed_factor):
        # jitter on the distance
        if abs(speed_factor) < 1:
            distance_factor = random.uniform(self.distance_jitter[0], self.distance_jitter[1])
        else:
            distance_factor = random.uniform(self.distance_jitter[0], 1)
        return distance_factor

    def frame_size_standardize(self, frame):
        """
        Standardize the frame size according to the settings in the cfg.
        Args:
            frame (Tensor): a single frame with the shape of (C, 1, H, W) to be
                standardized.
        """
        h,w,_ = frame.shape
        standard_size = self.cfg.PRETRAIN.STANDARD_SIZE
        if isinstance(standard_size, list):
            assert len(standard_size) == 3
            size_s, size_l, crop_size = standard_size
            reshape_size = random.randint(int(size_s), int(size_l))
        else:
            crop_size = standard_size
            reshape_size = standard_size

        # resize the short side to standard size
        dtype = frame.dtype
        frame = frame.permute(2, 0, 1).to(torch.float) # C, H, W
        aspect_ratio = random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        if h <= w:
            new_h = reshape_size
            new_w = int(new_h / h * w)
            # resize
            frame = F.resize(frame.unsqueeze(0), (new_h, new_w), "bilinear").squeeze(0) 
        elif h > w:
            new_w = reshape_size
            new_h = int(new_w / w * h)
            # resize
            frame = F.resize(frame.unsqueeze(0), (new_h, new_w), "bilinear").squeeze(0) 
            
        # crop
        if aspect_ratio >= 1: 
            crop_h = int(crop_size / aspect_ratio)
            crop_w = crop_size
        else:
            crop_h = crop_size
            crop_w = int(crop_size * aspect_ratio)
        start_h = random.randint(0, new_h - crop_h)
        start_w = random.randint(0, new_w - crop_w)
        return frame[:, start_h:start_h+crop_h, start_w:start_w+crop_w].to(dtype).permute(1, 2, 0) # H, W, C

    def static_mask(self, frames):
        """
        Applys static mask with random position and size to 
        the generated pseudo motion sequence
        Args: 
            frames (Tensor): shape of (C,T,H,W)
        Returns:
            frames (Tensor): masked frames.
        """
        c, t, h, w = frames.shape
        rand_t = random.randint(0, t-1)
        mask_size_ratio = random.uniform(self.mask_size_ratio[0], self.mask_size_ratio[1])
        mask_size_x, mask_size_y = [int(w*mask_size_ratio), int(h*mask_size_ratio)]
        start_x = random.randint(0, w-mask_size_x)
        start_y = random.randint(0, h-mask_size_y)
        frames_out = frames[:, rand_t].unsqueeze(1).expand(-1, t, -1, -1).clone()
        frames_out[:, :, start_y:start_y+mask_size_y, start_x:start_x+mask_size_x] = frames[
            :, :, start_y:start_y+mask_size_y, start_x:start_x+mask_size_x
        ]
        return frames_out

    def config_transform(self):
        """
        Configures the transformation applied to the pseudo motion sequence. 
        """
        std_transform_list = []
        if self.split == 'train' or self.split == 'val':
            # To tensor and normalize
            std_transform_list += [
                transforms.ToTensorVideo(),
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
                transforms.RandomHorizontalFlipVideo(),
            ]
            self.transform = Compose(std_transform_list)
        elif self.split == 'test':
            std_transform_list += [
                transforms.ToTensorVideo(),
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)

    def __call__(self, frames, index):
        return self.sample_generator(frames, index), self.labels
