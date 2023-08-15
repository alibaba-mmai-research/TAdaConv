#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""
Logging.
Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/logging.py.
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import builtins
import decimal
import functools
import logging
import os
import sys
import simplejson

import tadaconv.utils.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(cfg, log_file):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    if du.is_master_proc(du.get_world_size()):
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()
        return

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if du.is_master_proc(du.get_world_size()):
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if log_file is not None and du.is_master_proc(du.get_world_size()):
        filename = os.path.join(cfg.OUTPUT_DIR, log_file)
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("{:s}".format(json_stats))
