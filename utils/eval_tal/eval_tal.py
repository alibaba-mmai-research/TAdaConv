#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

import sys
from .eval_epic_detection import Epicdetection
from utils import logging
import numpy as np
import json
logger = logging.get_logger(__name__)


def evaluate_detection(video_anno, detection_result_file, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """
    Evaluate action detection performance.
    Args:
        video_anno (str): Annotation file path.
        detection_result_file (str): The detection results output by your model.
        tiou_thresholds (np.array): Iou thresholds to be tested.
    """
    detection = Epicdetection(video_anno, detection_result_file,
                                tiou_thresholds=tiou_thresholds,
                                subset='validation', verbose=True, check_status=False)
    detection.evaluate()
