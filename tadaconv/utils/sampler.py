#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Multi-fold distributed sampler."""

import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class MultiFoldDistributedSampler(Sampler):
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

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
    """

    def __init__(self, dataset, num_folds=1, num_replicas=None, rank=None, shuffle=True):
        """
            When num_folds = 1, MultiFoldDistributedSampler degenerates to DistributedSampler.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_folds = num_folds
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_folds * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = []
        for fold_idx in range(self.num_folds):
            g = torch.Generator()
            g.manual_seed(self.epoch+fold_idx)
            if self.shuffle:
                indices += torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices += list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch