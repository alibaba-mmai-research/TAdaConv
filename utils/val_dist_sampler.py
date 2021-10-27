#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Multi-fold distributed sampler."""

import math
import torch
import utils.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import utils.logging as logging
logger = logging.get_logger(__name__)


class MultiSegValDistributedSampler(DistributedSampler):
    """Modified from DistributedSampler, which performs multi fold training for 
    accelerating distributed training with large batches.
    
    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices

    .. warning::
        In distributed mode, calling the ``set_epoch`` method is needed to
        make shuffling work; each process will use the same random seed
        otherwise.

    Example::

        >>> sampler = MultiSegValDistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        """
            We divide each video in epic dataset into multiple sliding windows.
            Each sliding window is a sample in validation process for efficient.
            This function will assign the sliding windows which belong to the same video to a same gpu. 
        """
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        assert shuffle is False
        self.shuffle = shuffle
        vid_name_dict = {}
        self.vid_name_list = []
        self.vid_num_list = []
        for s in dataset._samples:
            if s[0] not in vid_name_dict:
                vid_name_dict[s[0]] = 0
                self.vid_name_list += [s[0]]
                self.vid_num_list += [0]
            self.vid_num_list[-1] += 1
        self.num_samples = int(math.ceil(len(self.vid_name_list) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.__init_dist__()

    def __init_dist__(self):
        indices = list(range(len(self.vid_name_list)))
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        self.true_indices = []
        for ind in indices:
            if ind == 0:
                exist_num = 0
            else:
                exist_num = sum(self.vid_num_list[:ind])
            self.true_indices.extend(list(range(exist_num, exist_num+self.vid_num_list[ind])))

    def __iter__(self):
        return iter(self.true_indices)

    def __len__(self):
        return len(self.true_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
