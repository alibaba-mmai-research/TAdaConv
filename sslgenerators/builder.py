#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Builder for self-supervised generator."""

from utils.registry import Registry

SSL_GENERATOR_REGISTRY = Registry("SSL_Methods")

def build_ssl_generator(cfg, split): 
    """
    Entry point to registered self-supervised learning methods. 
    Returns transformed frames and the self-supervised label.
    Args: 
        split (str): training, validation or test. 
    """
    ssl_generator = SSL_GENERATOR_REGISTRY.get(cfg.PRETRAIN.GENERATOR)(cfg, split)
    return ssl_generator
    