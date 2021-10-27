#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""
Functions that evaluate the temporal action localization performance for epic dataset.
Modified from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py. 
"""

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sys
from utils import logging
logger = logging.get_logger(__name__)
class Epicdetection(object):

    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=True,
                 assign_class=None,
                 classes=None):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.assign_class = assign_class
        if assign_class is not None:
            logger.info("assign_class:{} for detection".format(assign_class))
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.
        self.blocked_videos = list()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index, self.verb_all_label, self.noun_all_label = self._import_ground_truth(
            ground_truth_filename, classes)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            logger.info('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            logger.info('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            logger.info('\tNumber of predictions: {}'.format(nr_pred))
            logger.info('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))
        sys.stdout.flush()


    def _import_ground_truth(self, ground_truth_filename, classes=None):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        verb_all_label = {}
        noun_all_label = {}
        video_lst, t_start_lst, t_end_lst, label_lst, verb_lst, noun_lst = [], [], [], [], [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            '''
            if self.assign_class is not None and v['annotations'][0]['label'] != self.assign_class:
                continue
            '''
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    if classes is None:
                        activity_index[ann['label']] = cidx
                        cidx += 1
                    else:
                        activity_index[ann['label']] = classes.index(ann['label'])
                if int(ann['label'].split(',')[0]) not in verb_all_label:
                    verb_all_label[int(ann['label'].split(',')[0])] = 0
                if int(ann['label'].split(',')[1]) not in noun_all_label:
                    noun_all_label[int(ann['label'].split(',')[1])] = 0
                verb_lst.append(int(ann['label'].split(',')[0]))
                noun_lst.append(int(ann['label'].split(',')[1]))
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])
                

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst,
                                     'verb': verb_lst,
                                     'noun': noun_lst})
        return ground_truth, activity_index, verb_all_label, noun_all_label

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        verb_lst, noun_lst = [], []
        olabel_list = []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                if result['label'] not in self.activity_index:
                    continue
                label = self.activity_index[result['label']]
                '''
                if self.assign_class is not None and label != self.assign_class:
                    continue
                '''
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                olabel_list.append(result['label'])
                score_lst.append(result['score'])
                verb_lst.append(int(result['verb']))
                noun_lst.append(int(result['noun']))
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'olabel': olabel_list,
                                   'score': score_lst,
                                   'verb': verb_lst,
                                   'noun': noun_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label. 
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            # logger.info('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, label_name_list, group_name):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(label_name_list)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby(group_name)
        prediction_by_label = self.prediction.groupby(group_name)
        if self.assign_class is not None:
            set_trace()
            label_name = self.assign_class
            cidx = self.activity_index[label_name]
            compute_average_precision_detection(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
            )
        else:
            if 'label' in group_name:
                results = Parallel(n_jobs=32)(
                            delayed(compute_average_precision_detection)(
                                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                                tiou_thresholds=self.tiou_thresholds,
                            ) for cidx, label_name in enumerate(label_name_list))
            else:
                # for cidx, label_name in enumerate(label_name_list):
                #     compute_average_precision_detection(
                #                 ground_truth=ground_truth_by_label.get_group(label_name).reset_index(drop=True),
                #                 prediction=self._get_predictions_with_label(prediction_by_label, label_name, label_name),
                #                 tiou_thresholds=self.tiou_thresholds,
                #             )
                results = Parallel(n_jobs=16)(
                            delayed(compute_average_precision_detection)(
                                ground_truth=ground_truth_by_label.get_group(label_name).reset_index(drop=True),
                                prediction=self._get_predictions_with_label(prediction_by_label, label_name, label_name),
                                tiou_thresholds=self.tiou_thresholds,
                            ) for cidx, label_name in enumerate(label_name_list))
            for i in range(len(label_name_list)):
                ap[:,i] = results[i]

            return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap_action = self.wrapper_compute_average_precision(list(self.activity_index.keys()), 'label')
        self.print_map(self.ap_action, 'self.ap_action')

        self.ap_noun = self.wrapper_compute_average_precision(list(self.noun_all_label.keys()), 'noun')
        self.print_map(self.ap_noun, 'self.ap_noun')
        self.ap_verb = self.wrapper_compute_average_precision(list(self.verb_all_label.keys()), 'verb')
        self.print_map(self.ap_verb, 'self.ap_verb')

    def print_map(self, ap, _type):
        mAP = ap.mean(axis=1)
        average_mAP = mAP.mean()
        map_list = mAP.tolist()
        tiou_list = self.tiou_thresholds.tolist()
        map_str = ', '.join(["%.02f:%.04f"%(t, m) for t, m in zip(tiou_list, map_list)])
        logger.info(map_str)
        if self.verbose:
            logger.info('[RESULTS] Performance on ActivityNet detection task.')
            logger.info('\tAverage-mAP for {}: {}'.format(_type, average_mAP))
        sys.stdout.flush()

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        # set_trace()
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU