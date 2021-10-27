#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Wrapper for reading from OSS. """

import io
import os
import oss2 as oss
from utils import logging

logger = logging.get_logger(__name__)

def initialize_bucket(key, secret, endpoint, bucket, retries=10):
    """
    Wrapper for bucket initialization, with specified key, secret, endpoint, and bucket name.
    See more: https://pypi.org/project/oss2/. 
    Args:
        key      (string): The key to the account.
        secret   (string): The secret to the account.
        endpoint (string): The endpoint for the bucket.
        bucket   (string): The name of the bucket.
        retries  (int):    The number of retries for initializing the bucket.
    Returns:
        data_bucket (oss2.Bucket): The initialized bucket for accessing data.
    """
    for retry in range(retries):
        try:
            authentication = oss.Auth(key, secret)
            data_bucket = oss.Bucket(authentication, endpoint, bucket, connect_timeout=10)
            logger.info("OSS bucket [{}] initialized.".format(bucket))
            return data_bucket
        except:
            logger.info("OSS bucket [{}] initialization failed. Retrying... {}".format(bucket, retry))
    
    raise ValueError("OSS initialization failed. Please check your OSS connection.")

def read_from_buffer(bucket, oss_file, bucket_name, retries=10):
    """
    Wrapper for reading data directly to the memory of the local machine, from the specified bucket. 
    See more: https://pypi.org/project/oss2/. 
    Args:
        bucket      (oss2.Bucket):  Initialized Bucket object 
        oss_file    (string):       The name of the file on the oss to be read from. This should start with "oss://..."
        bucket_name (string):       The name of the bucket. 
                                    Here is how to find the bucket name "oss://{bucket_name}/...".
        retries     (int):          The number of retries for reading from the buffer.
    Returns:
        buf         (io.BytesIO):   The BytesIO object that can be directly read. 
    """
    for retry in range(retries):
        try:
            buf = io.BytesIO(bucket.get_object(oss_file.split(bucket_name)[-1][1:]).read())
            return buf
        except: 
            if retry < retries-1:
                logger.info("OSS download failed. {}/{} File: {}. Retrying...".format(
                    retry+1, retries, oss_file, 
                ))
            else:
                logger.info("OSS download failed. File: {}. Trying other videos.".format(
                    oss_file
                ))
    
    raise ValueError("OSS download failed. Please check your OSS connection. ")

def read_from_bucket(bucket, oss_file, local_file, bucket_name, retries=10):
    """
    Wrapper for reading data to the hard drive of local machine, from the specified bucket. 
    See more: https://pypi.org/project/oss2/. 
    Args:
        bucket      (oss2.Bucket):  Initialized Bucket object 
        oss_file    (string):       The name of the file on the oss to be read from. This should start with "oss://..."
        local_file  (string):       Place to store the downloaded file, which is also required to include the file name.
        bucket_name (string):       The name of the bucket. 
                                    Here is how to find the bucket name "oss://{bucket_name}/...".
        retries     (int):          The number of retries for downloading from the oss.
    """
    for i in range(retries):
        try:
            assert type(bucket) == oss.api.Bucket, TypeError("Input bucket should be type of {}".format(oss.api.Bucket))
            bucket.get_object_to_file(oss_file.split(bucket_name)[-1][1:], local_file)
            break
        except:
            if i == retries-1:
                logger.debug('Exceed maxmium tries for getting file {}'.format(
                    oss_file
                ))
    return True

def put_to_bucket(bucket, oss_file, local_file, bucket_name, retries=10):
    """
    Wrapper for putting data to the specified bucket.
    See more: https://pypi.org/project/oss2/. 
    Args:
        bucket      (oss2.Bucket):  Initialized Bucket object 
        oss_file    (string):       Where to put the data. This should start with "oss://{bucket_name}/..."
        local_file  (string):       The local file to be uploaded.
        bucket_name (string):       The name of the bucket. 
                                    Here is how to find the bucket name "oss://{bucket_name}/...".
        retries     (int):          The number of retries for putting to the bucket.
    """
    for i in range(retries):
        try:
            assert type(bucket) == oss.api.Bucket, TypeError("Input bucket should be type of {}".format(oss.api.Bucket))
            bucket.put_object_from_file(
                oss_file.split(bucket_name)[-1][1:] + local_file.split('/')[-1],
                local_file
            )
            logger.info("putting '{}' to '{}'".format(
                local_file,
                oss_file.split(bucket_name)[-1][1:] + local_file.split('/')[-1]
            ))
            break
        except:
            if i == retries-1:
                logger.debug('Exceed maxmium tries for getting file {}'.format(
                    oss_file
                ))

def clear_tmp_file(file_to_remove):
    """
    Remove the temporary files. 
    Args:
        file_to_remove (string or BytesIO): If given "string", file will be removed. 
                                            If given "BytesIO", the object will be closed.
    """
    for f in file_to_remove:
        if f is None:
            continue
        try:
            f.close()
        except:
            os.system('rm -rf {}'.format(f))