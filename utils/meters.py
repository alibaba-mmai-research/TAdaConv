#!/usr/bin/env python3
# Copyright (C) Alibaba Group H volding Limited. 

"""
Meters.
Modifed from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/meters.py.
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from utils.timer import Timer

import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.distributed as du

logger = logging.get_logger(__name__)

class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        cfg,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.cfg = cfg
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))

        self.video_labels = (
            torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.clip_indices = torch.linspace(0, num_videos-1, num_videos).long()
        self.model_ema_enabled = False
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter" if not self.model_ema_enabled else "ema_test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            # "{}: {}".format(i, k)
                            # for i, k in enumerate(self.clip_count.tolist())
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final" if not self.model_ema_enabled else "ema_test_final"}
        num_topks_correct = metrics.topks_correct(
            self.video_preds, self.video_labels, ks
        )
        topks = [
            (x / self.video_preds.size(0)) * 100.0
            for x in num_topks_correct
        ]
        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )
        logging.log_json_stats(stats)
    
    def set_model_ema_enabled(self, model_ema_enabled):
        self.model_ema_enabled = model_ema_enabled

class EpicKitchenMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.

    For the EpicKitchenMeter specifically, it caters to the need of the EpicKitchens
    dataset, where both verbs and nouns are predicted before actions are predicted using
    those predictions.
    """

    def __init__(
        self,
        cfg,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            cfg (Config): the global config object.
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.cfg = cfg
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.num_videos = num_videos
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method

        assert self.ensemble_method in ["sum", "max"], f"Ensemble Method {ensemble_method} is not supported"
        
        if cfg.DATA.MULTI_LABEL or not hasattr(cfg.DATA, "TRAIN_VERSION"):
            # Initialize tensors.
            self.video_preds = {
                "verb_class": torch.zeros((num_videos, self.num_clips, num_cls[0])),
                "noun_class": torch.zeros((num_videos, self.num_clips, num_cls[1])),
                "action_class_ind_pred": torch.zeros((num_videos, self.num_clips, num_cls[0]*num_cls[1]))
            }

            self.video_labels = {
                "verb_class": torch.zeros((num_videos)),  # verb
                "noun_class": torch.zeros((num_videos)),  # noun
                "action_class_ind_pred": torch.zeros((num_videos)),
            }
            self.update_stats = self.update_stats_multi_label
            self.finalize_metrics = self.finalize_metrics_multi_label
        elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION in ["only_train_verb", "only_train_noun"]:
            self.video_preds = torch.zeros((num_videos, self.num_clips, num_cls))
            self.video_labels = torch.zeros((num_videos))
            self.update_stats = self.update_stats_separate_label
            self.finalize_metrics = self.finalize_metrics_separate_label
        else: raise NotImplementedError
        self.video_names = {i: "" for i in range(num_videos)}
        self.clip_count = torch.zeros((num_videos)).long()
        self.clip_indices = torch.linspace(0, num_videos-1, num_videos).long()
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        if isinstance(self.video_preds, dict):
            for k, v in self.video_preds.items():
                v.zero_()
            for k, v in self.video_labels.items():
                v.zero_()
        else:
            self.video_preds.zero_()
            self.video_labels.zero_()

    def update_stats_separate_label(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble, for separate verb and noun training.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            view_id = int(clip_ids[ind]) % self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )

            self.video_labels[vid_id] = labels[ind]

            self.video_preds[vid_id][view_id] = preds[ind]

            self.clip_count[vid_id] += 1

    def update_stats_multi_label(self, preds_verb, preds_noun, labels_verb, labels_noun, clip_ids, names=[]):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble, for joint verb and noun training.
        Args:
            preds_verb (tensor): verb predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls[0]).
            preds_noun (tensor): noun predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls[1]).
            labels_verb (tensor): the corresponding verb labels of the current batch.
                Dimension is N.
            labels_noun (tensor): the corresponding noun labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
            names (list): list of video names.
        """
        for ind in range(preds_verb.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            view_id = int(clip_ids[ind]) % self.num_clips
            if self.video_labels["verb_class"][vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels["verb_class"][vid_id].type(torch.FloatTensor),
                    labels_verb[ind].type(torch.FloatTensor),
                )
                assert torch.equal(
                    self.video_labels["noun_class"][vid_id].type(torch.FloatTensor),
                    labels_noun[ind].type(torch.FloatTensor),
                )
            if len(names) > 0:
                if self.video_names[vid_id] != "":
                    assert self.video_names[vid_id] == names[ind], \
                        f"For {vid_id}, its name {self.video_names[vid_id]} should be equal to {names[ind]}"
                else:
                    self.video_names[vid_id] = names[ind]

            self.video_labels["verb_class"][vid_id] = labels_verb[ind]
            self.video_labels["noun_class"][vid_id] = labels_noun[ind]
            self.video_labels["action_class_ind_pred"][vid_id] = labels_verb[ind] * preds_noun.shape[1] + labels_noun[ind]

            self.video_preds["verb_class"][vid_id][view_id] = preds_verb[ind]
            self.video_preds["noun_class"][vid_id][view_id] = preds_noun[ind]
            self.video_preds["action_class_ind_pred"][vid_id][view_id] = (preds_verb[ind].unsqueeze(-1) * preds_noun[ind].unsqueeze(-2)).reshape(-1)

            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter" if not self.model_ema_enabled else "ema_test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics_multi_label(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics for joint verb and 
        noun training.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            # "{}: {}".format(i, k)
                            # for i, k in enumerate(self.clip_count.tolist())
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final" if not self.model_ema_enabled else "ema_test_final"}
        video_preds = {}
        if self.ensemble_method == "sum":
            video_preds["verb_class"] = self.video_preds["verb_class"].sum(1)
            video_preds["noun_class"] = self.video_preds["noun_class"].sum(1)
            video_preds["action_class_ind_pred"] = self.video_preds["action_class_ind_pred"].sum(1)
        elif self.ensemble_method == "max":
            video_preds["verb_class"] = self.video_preds["verb_class"].max(1)[0]
            video_preds["noun_class"] = self.video_preds["noun_class"].max(1)[0]
            video_preds["action_class_ind_pred"] = self.video_preds["action_class_ind_pred"].max(1)[0]
        num_topks_correct, b = metrics.joint_topks_correct(
            video_preds, self.video_labels, ks
        )
        for name, v in num_topks_correct.items():
            topks = [ (x / b) * 100.0 for x in v ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                stats["top_{}_acc_{}".format(name, k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        logging.log_json_stats(stats)

    def finalize_metrics_separate_label(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics, for separate verb 
        and noun training.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final" if not self.model_ema_enabled else "ema_test_final"}
        if self.ensemble_method == "sum":
            video_preds = self.video_preds.sum(1)
        elif self.ensemble_method == "max":
            video_preds = self.video_preds.max(1)[0]
        num_topks_correct = metrics.topks_correct(
            video_preds, self.video_labels, ks
        )
        topks = [
            (x / self.video_preds.size(0)) * 100.0
            for x in num_topks_correct
        ]
        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )
        logging.log_json_stats(stats)

    def set_model_ema_enabled(self, model_ema_enabled):
        """
        Whether the meter logs for ema models or not.
        Args:
            model_ema_enabled (bool): indicator of whether ema model 
                is enabled.
        """
        self.model_ema_enabled = model_ema_enabled

    def get_video_preds(self):
        """
        Returns the saved video predictions.
        """
        video_preds = {}
        if self.ensemble_method == "sum":
            video_preds["verb_class"] = self.video_preds["verb_class"].sum(1)
            video_preds["noun_class"] = self.video_preds["noun_class"].sum(1)
            video_preds["action_class_ind_pred"] = self.video_preds["action_class_ind_pred"].sum(1)
        elif self.ensemble_method == "max":
            video_preds["verb_class"] = self.video_preds["verb_class"].max(1)[0]
            video_preds["noun_class"] = self.video_preds["noun_class"].max(1)[0]
            video_preds["action_class_ind_pred"] = self.video_preds["action_class_ind_pred"].max(1)[0]
        return video_preds

class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size=10):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (Config): the global config object.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.OPTIMIZER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.opts = defaultdict(ScalarMeter)

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.opts = defaultdict(ScalarMeter)
        

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size, **kwargs):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.opts[k].add_value(v)

        if not self._cfg.PRETRAIN.ENABLE and not self._cfg.LOCALIZATION.ENABLE:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size
    def update_custom_stats(self, stats):
        """
        Update stats using custom keys.
        Args:
            stats (dict): additional stats to be updated.
        """
        for k,v in stats.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.opts[k].add_value(v)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.OPTIMIZER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            # "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        for k,v in self.opts.items():
            stats[k] = v.get_win_median()
        if not self._cfg.PRETRAIN.ENABLE and not self._cfg.LOCALIZATION.ENABLE:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.OPTIMIZER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }
        for k,v in self.opts.items():
            stats[k] = v.get_global_avg()
        if not self._cfg.PRETRAIN.ENABLE:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (Config): the global config object.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.model_ema_enabled = False
        self.opts = defaultdict(ScalarMeter)

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.opts = defaultdict(ScalarMeter)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size, **kwargs):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.opts[k].add_value(v)
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size
    
    def update_custom_stats(self, stats):
        """
        Update stats using custom keys.
        Args:
            stats (dict): additional stats to be updated.
        """
        for k,v in stats.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.opts[k].add_value(v)
    
    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter" if not self.model_ema_enabled else "ema_val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.OPTIMIZER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        for k,v in self.opts.items():
            stats[k] = v.get_win_median()
        stats["top1_err"] = self.mb_top1_err.get_win_median()
        stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch" if not self.model_ema_enabled else "ema_val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.OPTIMIZER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }
        for k,v in self.opts.items():
            if "top1_err" in k or "top5_err" in k:
                stats[k] = v.get_win_median()
            else:
                stats[k] = v.get_global_avg()
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        self.min_top5_err = min(self.min_top5_err, top5_err)

        stats["top1_err"] = top1_err
        stats["top5_err"] = top5_err
        stats["min_top1_err"] = self.min_top1_err
        stats["min_top5_err"] = self.min_top5_err

        logging.log_json_stats(stats)
    
    def set_model_ema_enabled(self, model_ema_enabled):
        self.model_ema_enabled = model_ema_enabled