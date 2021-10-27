#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" 
Losses for temporal action localization . 
Modified from https://github.com/JJBOY/BMN-Boundary-Matching-Network/blob/master/loss_function.py.
"""

import torch
import numpy as np
import torch.nn.functional as F
from utils.registry import Registry
LOCALIZATION_LOSSES = Registry("Localization_Losses")


@LOCALIZATION_LOSSES.register()
def Loss_Tem(cfg, preds, logits, labels={}, cur_epoch=0):
    """
    Calculate start and end loss.
    Args:
        preds (dict): predicted start and end sequences.
        logits (Tensor): Only for placeholders, no use.
        labels (Tensor): start and end sequences label.
    """
    pred_start = preds['start']
    pred_end = preds['end']
    gt_start = labels['supervised']['start_map']
    gt_end = labels['supervised']['end_map']
    label_weight = torch.ones(pred_start.shape[0], device=pred_start.device)
    def bi_loss(pred_score, gt_label, label_weight):
        label_weight = label_weight.unsqueeze(1).expand_as(pred_score).reshape(-1)
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float() * label_weight
        num_entries = label_weight.sum()
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask * label_weight
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask) * label_weight
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start, label_weight)
    loss_end = bi_loss(pred_end, gt_end, label_weight)
    loss = loss_start + loss_end
    return {"tem": loss}, None


@LOCALIZATION_LOSSES.register()
def Loss_BmnActionCls(cfg, preds, logits, labels={}, cur_epoch=0):
    """
    Calculate action classification loss for proposals, but this donot work in epic dataset.
    Args:
        preds (dict): predicted action classification maps.
        logits (Tensor): Only for placeholders, no use.
        labels (Tensor): classification maps label.
    """
    b, c, _, _ = labels['supervised']['label_map'].shape
    gt_label = labels['supervised']['label_map'].flatten(2, 3)
    gt_iou_map = (labels['supervised']['iou_map'] * labels['supervised']['mask']).flatten(1, 2)
    verb_map = preds['verb_map'].flatten(2, 3)
    noun_map = preds['noun_map'].flatten(2, 3)
    select_action = gt_iou_map >= 0.75

    select_action = select_action.view(-1)
    gt_label = gt_label.permute(0, 2, 1).flatten(0, 1)[select_action, :]
    verb_map = verb_map.permute(0, 2, 1).flatten(0, 1)[select_action, :]
    noun_map = noun_map.permute(0, 2, 1).flatten(0, 1)[select_action, :]
    verb_loss = F.cross_entropy(verb_map, gt_label[:, 0])
    noun_loss = F.cross_entropy(noun_map, gt_label[:, 1])
    
    return {"verb_loss": verb_loss, "noun_loss": noun_loss}, None

@LOCALIZATION_LOSSES.register()
def Loss_PemReg(cfg, preds, logits, labels={}, cur_epoch=0):
    """
    Regression confidence maps.
    Args:
        preds (dict): predicted regression confidence maps.
        logits (Tensor): Only for placeholders, no use.
        labels (Tensor): iou maps for label.
    """
    pred_score = preds['confidence_map'][:, 0]
    gt_iou_map = labels['supervised']['iou_map']
    mask = labels['supervised']['mask']
    gt_iou_map = gt_iou_map * mask

    u_hmask = (gt_iou_map > cfg.LOCALIZATION.POS_REG_THRES).float()
    u_mmask = ((gt_iou_map <= cfg.LOCALIZATION.POS_REG_THRES) & (gt_iou_map > cfg.LOCALIZATION.NEG_REG_THRES)).float()
    u_lmask = ((gt_iou_map <= cfg.LOCALIZATION.NEG_REG_THRES) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)
    if num_m == 0:
        r_m = num_h / (num_m+1)
    else:
        r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    weights = u_hmask + u_smmask + u_slmask

    loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
    loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)
    if torch.isnan(loss):
        stop = 1
    return {"pem_reg": loss}, None

@LOCALIZATION_LOSSES.register()
def Loss_PemCls(cfg, preds, logits, labels={}, cur_epoch=0):
    """
    Binary classification confidence maps.
    Args:
        preds (dict): predicted classification confidence maps.
        logits (Tensor): Only for placeholders, no use.
        labels (Tensor): iou maps for label.
    """
    pred_score = preds['confidence_map'][:, 1]
    gt_iou_map = labels['supervised']['iou_map']
    mask = labels['supervised']['mask']
    gt_iou_map = gt_iou_map * mask
    
    pmask = (gt_iou_map > cfg.LOCALIZATION.POS_CLS_THRES).float()
    nmask = (gt_iou_map <= cfg.LOCALIZATION.POS_CLS_THRES).float()
    nmask = nmask * mask

    num_positive = torch.sum(pmask)
    num_entries = num_positive + torch.sum(nmask)
    if num_positive == 0:
        ratio = 0.0
    else:
        ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask 
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask 
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    if torch.isnan(loss):
        stop = 1
    return {"pem_cls": loss}, None


