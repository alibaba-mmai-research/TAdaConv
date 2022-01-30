#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Entry file for training, evaluating and testing a video model."""

import os
import sys
import time
sys.path.append(os.path.abspath(os.curdir))

from utils.launcher import launch_task

from test import test
from train import train
from test_epic_localization import test_epic_localization
from submission_test import submission_test

from utils.config import Config

def _prepare_data(cfg):
    if cfg.TASK_TYPE in ['classification']:
        train_func = train
        test_func = test
    elif cfg.TASK_TYPE in ['localization']:
        train_func = train
        test_func = test_epic_localization
    elif cfg.TASK_TYPE in ["submission"]:
        cfg.TRAIN.ENABLE = False
        cfg.TEST.ENABLE = False
        train_func = None
        test_func = None
        submission_func = submission_test
    else:
        raise ValueError("unknown TASK_TYPE {}".format(cfg.TASK_TYPE))
    
    run_list = []
    if cfg.TRAIN.ENABLE:
        # Training process is performed by the entry function defined above.
        run_list.append([cfg.deep_copy(), train_func])
    
    if cfg.TEST.ENABLE:
        # Test is performed by the entry function defined above.
        run_list.append([cfg.deep_copy(), test_func])
        if cfg.TEST.AUTOMATIC_MULTI_SCALE_TEST:
            """
                By default, test_func performs single view test. 
                AUTOMATIC_MULTI_SCALE_TEST automatically performs multi-view test after the single view test.
            """
            cfg.LOG_MODEL_INFO = False
            cfg.LOG_CONFIG_INFO = False

            cfg.TEST.NUM_ENSEMBLE_VIEWS = 10
            cfg.TEST.NUM_SPATIAL_CROPS = 1

            if "kinetics" in cfg.TEST.DATASET or "epickitchen" in cfg.TEST.DATASET:
                cfg.TEST.NUM_SPATIAL_CROPS = 3
            if "imagenet" in cfg.TEST.DATASET and not cfg.PRETRAIN.ENABLE:
                cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
                cfg.TEST.NUM_SPATIAL_CROPS = 3
            if "ssv2" in cfg.TEST.DATASET:
                cfg.TEST.NUM_ENSEMBLE_VIEWS = 2
                cfg.TEST.NUM_SPATIAL_CROPS = 3
            cfg.TEST.LOG_FILE = "val_{}clipsx{}crops.log".format(
                cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
            )
            run_list.append([cfg.deep_copy(), test_func])

    if cfg.SUBMISSION.ENABLE:
        # currently only supports epic kitchen submission
        cfg.LOG_MODEL_INFO = False
        cfg.TEST.NUM_ENSEMBLE_VIEWS = 10
        cfg.TEST.NUM_SPATIAL_CROPS = 3

        cfg.TEST.LOG_FILE = "test_{}clipsx{}crops.log".format(
            cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
        )
        run_list.append([cfg.deep_copy(), submission_func])
  
    return run_list

def main():
    """
    Entry function for spawning all the function processes. 
    """
    cfg = Config(load=True)

    # get the list of configs and functions for running
    run_list = _prepare_data(cfg)

    for run in run_list:
        launch_task(cfg=run[0], init_method=run[0].get_args().init_method, func=run[1])

    print("Finish running with config: {}".format(cfg.args.cfg_file))


if __name__ == "__main__":
    main()
