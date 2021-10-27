#!/usr/bin/env python3

""" Task launcher. """

import os
import torch
from utils.misc import get_num_gpus

def launch_task(cfg, init_method, func):
    """
    Launches the task "func" on one or multiple devices.
    Args:
        cfg (Config): global config object. 
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): task to run.
    """
    torch.cuda.empty_cache()
    if get_num_gpus(cfg) > 1:
        if cfg.PAI:
            # if using the PAI cluster, get info from the environment
            cfg.SHARD_ID = int(os.environ['RANK'])
            if "VISIBLE_DEVICE_LIST" in os.environ:
                cfg.NUM_GPUS = len(os.environ["VISIBLE_DEVICE_LIST"].split(","))
            else:
                cfg.NUM_GPUS = torch.cuda.device_count()
            cfg.NUM_SHARDS = int(os.environ['WORLD_SIZE'])

        torch.multiprocessing.spawn(
            run,
            nprocs=cfg.NUM_GPUS,
            args=(func, init_method, cfg),
            daemon=False,
        )
    else:
        func(cfg=cfg)

def run(
    local_rank, func, init_method, cfg
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
        cfg (Config): global config object.
    """

    num_proc    = cfg.NUM_GPUS      # number of nodes per machine
    shard_id    = cfg.SHARD_ID
    num_shards  = cfg.NUM_SHARDS    # number of machines
    backend     = cfg.DIST_BACKEND  # distribued backends ('nccl', 'gloo' or 'mpi')

    world_size  = num_proc * num_shards
    rank        = shard_id * num_proc + local_rank
    cfg.LOCAL_RANK = rank

    # dump machine info
    print("num_proc (NUM_GPU): {}".format(num_proc))
    print("shard_id (os.environ['RANK']): {}".format(shard_id))
    print("num_shards (os.environ['WORLD_SIZE']): {}".format(num_shards))
    print("rank: {}".format(rank))
    print("local_rank (GPU_ID): {}".format(local_rank))

    try:
        if cfg.PAI == False:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
        else:
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
            )
    except Exception as e:
        raise e
    
    if "VISIBLE_DEVICE_LIST" in os.environ:
        torch.cuda.set_device(int(os.environ["VISIBLE_DEVICE_LIST"]))
    else:
        torch.cuda.set_device(f'cuda:{local_rank}')
    os.system(f"CUDA_VISIBLE_DEVICES={local_rank}")
    func(cfg)
