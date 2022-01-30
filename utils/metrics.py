#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""
Functions for computing metrics.
Modifed from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/metrics.py.
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import torch

def joint_topks_correct(preds, labels, ks):
    """
    Calculates number of correctly predicted samples for each top-k value
    respectively for separate verb/noun, and joint action predictions.
    Args:
        preds (dict): dictionary of verb and noun predictions. can have 
            two keys "verb_class" and "noun_class", or alternatively
            three keys, "verb_class", "noun_class" and "action_class_ind_pred".
        labels (dict): dictionray of verb and noun class labels. The rules for
            the keys are the same as the preds.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        topks_correct_all (dict): number of top-k correctly predicted samples 
            for "verb_class", "noun_class", "action_class_ind_pred", 
            and "joint_class".
            The difference in the "action_class_ind_pred" and the "join_class" 
            is the sequence of calculating action score and fusing different 
            views. 
            Details can be found in the tech report, 
            Huang et al. 
            Towards Training Stronger Video Vision Transformers for 
            EPIC-KITCHENS-100 Action Recognition.
            https://arxiv.org/pdf/2106.05058.pdf
        b (int): batch size.
    """

    assert len(preds.keys()) <= 3, "Only a maximum of three joint topks are supported."
    for k in preds.keys():
        assert k in labels.keys(), "Predicted key not in labels."

    topks_correct_all = {}
    idx = 0
    joint_label = [0, 0, 0]
    num_classes = [0, 0, 0]
    for k, pred in preds.items():
        b = pred.shape[0]
        label = labels[k]
        joint_label[idx] = label
        num_classes[idx] = pred.shape[1]
        if idx == 0:
            if pred[0].sum(-1) != 1:
                pred = pred.softmax(-1)
            joint_pred = pred.unsqueeze(-1)
            idx += 1
        elif idx == 1:
            if pred[0].sum(-1) != 1:
                pred = pred.softmax(-1)
            joint_pred = joint_pred * pred.unsqueeze(-2)
            idx += 1

        assert pred.size(0) == label.size(0), "Batch dim of predictions and labels must match"
        _top_max_k_vals, top_max_k_inds = torch.topk(
            pred, max(ks), dim=1, largest=True, sorted=True
        )
        # (batch_size, max_k) -> (max_k, batch_size).
        top_max_k_inds = top_max_k_inds.t()
        # (batch_size, ) -> (max_k, batch_size).
        rep_max_k_labels = label.view(1, -1).expand_as(top_max_k_inds)
        # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
        top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
        # Compute the number of topk correct predictions for each k.
        topks_correct = [
            top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
        ]
        topks_correct_all[k] = topks_correct
    
    joint_pred = joint_pred.reshape(joint_pred.shape[0], -1)
    joint_label = joint_label[0] * num_classes[1] + joint_label[1]
    _top_max_k_vals, top_max_k_inds = torch.topk(
        joint_pred, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = joint_label.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    topks_correct_all["joint_class"] = topks_correct
    
    return topks_correct_all, b
    
    

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]