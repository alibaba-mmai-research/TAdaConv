#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Runs submission split with the trained video classification model."""

import numpy as np
import os
import pickle
import torch
import json

import utils.bucket as bu
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets.base.builder import build_loader
from models.base.builder import build_model
from utils.meters import TestMeter, EpicKitchenMeter

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_submission_test(test_loader, model, submission_meter, cfg):
    """
    Performs multi-view test on the submission set, where {cfg.TEST.NUM_ENSEMBLE_VIEWS}
    clips and {cfg.TEST.NUM_SPATIAL_CROPS} crops are sampled temporally and spatially, forming 
    in total cfg.TEST.NUM_ENSEMBLE_VIEWS x cfg.TEST.NUM_SPATIAL_CROPS views.
    The softmax scores are aggregated according to the {cfg.SUBMISSION.ACTION_CLASS_ENSUMBLE_METHOD}.
    The predictions are then organized into a dictionary before writing in the specified file. 
    Args:
        test_loader (loader): video testing loader for the submission set.
        model (model): the pretrained video model to perform test on the submission set.
        submission_meter (EpicKitchenMeter): epic kitchen submission meters to log and ensemble the testing
            results.
        cfg (Config): The global config object.
    """
    # Enable eval mode.
    model.eval()
    submission_meter.iter_tic()
    res_dic = {}
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            for k, v in inputs.items():
                if k == "sentences" or k == "name":
                    continue
                if isinstance(inputs[k], list):
                    for i in range(len(inputs[k])):
                        inputs[k][i] = v[i].cuda(non_blocking=True)
                else:
                    inputs[k] = v.cuda(non_blocking=True)

            # Transfer the labels to the current GPU device.
            if isinstance(labels["supervised"], dict):
                for k, v in labels["supervised"].items():
                    labels["supervised"][k] = v.cuda()
            else:
                labels["supervised"] = labels["supervised"].cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Perform the forward pass.
        preds, _ = model(inputs)

        if isinstance(labels["supervised"], dict):
            # Gather all the predictions across all the devices to perform ensemble.
            if misc.get_num_gpus(cfg) > 1:
                preds_verb, preds_noun, labels_verb, labels_noun, video_idx = du.all_gather(
                    [
                        preds["verb_class"], 
                        preds["noun_class"], 
                        labels["supervised"]["verb_class"], 
                        labels["supervised"]["noun_class"], 
                        video_idx
                    ]
                )
            else:
                preds_verb  = preds["verb_class"]
                preds_noun  = preds["noun_class"]
                labels_verb = labels["supervised"]["verb_class"]
                labels_noun = labels["supervised"]["noun_class"]
            if misc.get_num_gpus(cfg):
                preds_verb  = preds_verb.cpu()
                preds_noun  = preds_noun.cpu()
                labels_verb = labels_verb.cpu()
                labels_noun = labels_noun.cpu()
                video_idx   = video_idx.cpu()

            submission_meter.iter_toc()
            # Update and log stats.
            submission_meter.update_stats(
                preds_verb.detach(),
                preds_noun.detach(),
                labels_verb.detach(),
                labels_noun.detach(),
                video_idx.detach(),
                [test_loader.dataset._get_sample_info(i)["name"] for i in video_idx.tolist()] if "name" in test_loader.dataset._get_sample_info(0).keys() else []
            )
            submission_meter.log_iter_stats(cur_iter)
        else:

            # Gather all the predictions across all the devices to perform ensemble.
            if misc.get_num_gpus(cfg) > 1:
                preds, labels_supervised, video_idx = du.all_gather(
                    [preds, labels["supervised"], video_idx]
                )
            else:
                labels_supervised = labels["supervised"]
            if misc.get_num_gpus(cfg):
                preds = preds.cpu()
                labels_supervised = labels_supervised.cpu()
                video_idx = video_idx.cpu()

            submission_meter.iter_toc()
            # Update and log stats.
            submission_meter.update_stats(
                preds.detach(), labels_supervised.detach(), video_idx.detach()
            )
            submission_meter.log_iter_stats(cur_iter)
        submission_meter.iter_tic()

    if cfg.DATA.MULTI_LABEL or not hasattr(cfg.DATA, "TRAIN_VERSION"):
        video_preds = submission_meter.get_video_preds()

        if cfg.SUBMISSION.ACTION_CLASS_ENSUMBLE_METHOD == "calculate":
            action_class = (video_preds["verb_class"].unsqueeze(-1) * video_preds["noun_class"].unsqueeze(1))
            action_class = action_class.reshape(action_class.shape[0], -1)
        elif cfg.SUBMISSION.ACTION_CLASS_ENSUMBLE_METHOD == "sum":
            action_class = video_preds["action_class_ind_pred"]
        action_ind_top100 = action_class.topk(100)[1]

        results_dict = {
            "version": "0.2",
            "challenge": "action_recognition",
            "sls_pt": 2,
            "sls_tl": 3,
            "sls_td": 3,
            "results":{
                f"{submission_meter.video_names[vid_ind]}": {
                    "verb": {
                        f"{verb_ind}": video_preds["verb_class"][vid_ind][verb_ind].item() \
                            for verb_ind in range(video_preds["verb_class"].shape[1])
                    }, 
                    "noun": {
                        f"{noun_ind}": video_preds["noun_class"][vid_ind][noun_ind].item() \
                            for noun_ind in range(video_preds["noun_class"].shape[1])
                    },
                    "action": {
                        f"{action_ind//300},{action_ind%300}": action_class[vid_ind][action_ind].item() \
                            for action_ind in action_ind_top100[vid_ind].tolist()
                    }
                } for vid_ind in range(video_preds["verb_class"].shape[0])
            }
        }

        save_path = os.path.join(cfg.OUTPUT_DIR, cfg.SUBMISSION.SAVE_RESULTS_PATH)
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(
            "Successfully saved prediction results to {}.".format(save_path)
        )

        verb = submission_meter.video_preds["verb_class"]
        noun = submission_meter.video_preds["noun_class"]
        torch.save(verb, os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb.pyth"))
        torch.save(noun, os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun.pyth"))
        logger.info(
            "Successfully saved verb and noun results to {} and {}.".format(os.path.join(cfg.OUTPUT_DIR, "verb.pyth"), os.path.join(cfg.OUTPUT_DIR, "noun.pyth"))
        )
    elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION == "only_train_verb":
        verb = submission_meter.video_preds
        torch.save(verb, os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb.pyth"))
        logger.info(
            "Successfully saved verb results to {}.".format(os.path.join(cfg.OUTPUT_DIR, "verb.pyth"))
        )
    elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION == "only_train_noun":
        noun = submission_meter.video_preds
        torch.save(noun, os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun.pyth"))
        logger.info(
            "Successfully saved noun results to {}.".format(os.path.join(cfg.OUTPUT_DIR, "noun.pyth"))
        )


    submission_meter.finalize_metrics()
    submission_meter.reset()

def submission_test(cfg):
    """
    Performs multi-view test on the submission set, and save the prediction results.
    Currently only support EPIC-KITCHENS submission set.
    Args:
        cfg (Config): The global config object.
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Submission with config:")
        logger.info(cfg)

    # Build the video model and print model statistics.
    model, model_ema = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
    else:
        model_bucket = None

    cu.load_test_checkpoint(cfg, model, model_ema, model_bucket)

    # Create video testing loaders on the submission set.
    test_loader = build_loader(cfg, "submission")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing and results saving.
    cfg.LOG_PERIOD = max(len(test_loader) // 10, 5)
    submission_meter = EpicKitchenMeter(
        cfg,
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.VIDEO.HEAD.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Perform multi-view test on the submission set.
    submission_meter.set_model_ema_enabled(False)
    perform_submission_test(test_loader, model, submission_meter, cfg)
    
    # upload results to bucket
    if model_bucket is not None and du.is_master_proc():
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.SUBMISSION.SAVE_RESULTS_PATH)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
