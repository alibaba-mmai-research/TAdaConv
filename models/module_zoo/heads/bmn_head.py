#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Boundary Matching Network. 
Modified from https://github.com/JJBOY/BMN-Boundary-Matching-Network/blob/master/models.py.
"""


import os
import abc
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import Registry
from models.base.base_blocks import HEAD_REGISTRY

@HEAD_REGISTRY.register()
class BaseBMN(nn.Module):
    """
    Head for predicting boundary matching map.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): the global config object. 
        """
        super(BaseBMN, self).__init__()
        self.cfg = cfg
        self.tscale = cfg.DATA.TEMPORAL_SCALE
        self.dscale = cfg.DATA.DURATION_SCALE if cfg.DATA.DURATION_SCALE > 0 else cfg.DATA.TEMPORAL_SCALE
        self.num_sample = cfg.VIDEO.HEAD.NUM_SAMPLE
        self.num_sample_perbin = cfg.VIDEO.HEAD.NUM_SAMPLE_PERBIN
        self.hidden_dim_1d = cfg.VIDEO.DIM1D
        self.hidden_dim_2d = cfg.VIDEO.DIM2D
        self.hidden_dim_3d = cfg.VIDEO.DIM3D
        self.prop_boundary_ratio = cfg.VIDEO.HEAD.BOUNDARY_RATIO
        self._construct_head()

    def _construct_head(
        self,
    ):
        self.sample_mask = nn.Parameter(self.get_interp1d_mask(self.prop_boundary_ratio, self.num_sample), requires_grad=False)
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )
        if self.cfg.VIDEO.HEAD.USE_BMN_REGRESSION:
            self.x_2d_r = nn.Sequential(
                nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            )
        if type(self.cfg.VIDEO.HEAD.NUM_CLASSES) is list:
            self.x_2d_verb = nn.Sequential(
                nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, self.cfg.VIDEO.HEAD.NUM_CLASSES[0], kernel_size=1),
            )
            self.x_2d_noun = nn.Sequential(
                nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim_2d, self.cfg.VIDEO.HEAD.NUM_CLASSES[1], kernel_size=1),
            )

    def forward(self, x):
        """
        Args:
            x (dict): {
                "video" (tensor): Features for sliding windows.
            }
        Returns:
            output (dict): {
                confidence_map: (tensor),
                 start_map: (tensor),
                 end_map: (tensor),
                 reg_map: (tensor),
                 verb_map: (tensor),
                 noun_map: (tensor)
            } 
        """
        base_feature = x['video']
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        mid_feature = self.x_1d_p(base_feature)
        mid_feature = self._boundary_matching_layer(mid_feature)
        mid_feature = self.x_3d_p(mid_feature).squeeze(2)
        confidence_map = self.x_2d_p(mid_feature)
        if self.cfg.VIDEO.HEAD.USE_BMN_REGRESSION:
            reg_map = self.x_2d_r(mid_feature)
        else:
            reg_map = {}
        if hasattr(self, 'x_2d_verb'):
            verb_map = self.x_2d_verb(mid_feature)
            noun_map = self.x_2d_noun(mid_feature)
        else:
            verb_map, noun_map = {}, {}
        output = {"confidence_map": confidence_map,
                "start": start,
                "end": end,
                "reg_map": reg_map, 
                "verb_map": verb_map,
                "noun_map": noun_map}
        return output, {}

    def _boundary_matching_layer(self, x):
        """
        Apply boundary mathcing operation for input feature
        Args:
            x (tensor): 1D feature for boundary mathcing operation.
        Returns:
            output (Tensor): matched features for proposals
        """
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.dscale,self.tscale)
        return out

    def get_interp1d_mask(self, prop_boundary_ratio, num_sample):
        """
        generate sample mask for each point in Boundary-Matching Map
        Args:
            prop_boundary_ratio (float): Boundary expand ratio.
            num_sample (int): The number of sample points for each proposal.
        Returns:
            output (Tensor): sample mask
        """
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.dscale):
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        return torch.Tensor(mask_mat).view(self.tscale, -1)

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        """
        generate sample mask for a boundary-matching pair
        Args:
            seg_xmin (float): Start time of the proposal.
            seg_xmax (float): End time of the proposal.
            tscale (int): Temporal len for bmn.
            num_sample (int): The number of sample points for each proposal.
            num_sample_perbin (int): The number of sample points for each bin.
        Returns:
            output (Tensor): one sample mask
        """
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask
