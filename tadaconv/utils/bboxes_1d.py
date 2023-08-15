import numpy as np


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """
    calculate the overlap proportion between the anchor and all bbox for supervise signal,
    Args:
        anchors_min (np.ndarry): 1d anchors start position, shape is N.
        anchors_max (np.ndarry): 1d anchors end position, shape: N.
        box_min (np.ndarry): 1d boxes start position, shape: N.
        box_max (np.ndarry): 1d boxes end position, shape: N.
    Returns:
        scores: (np.ndarry)
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """
    Compute jaccard score between a box and the anchors.
    Args:
        anchors_min (np.ndarry): 1d anchors start position, shape is N.
        anchors_max (np.ndarry): 1d anchors end position, shape: N.
        box_min (np.ndarry): 1d boxes start position, shape: N.
        box_max (np.ndarry): 1d boxes end position, shape: N.
    Returns:
        jaccard: (np.ndarry)
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard