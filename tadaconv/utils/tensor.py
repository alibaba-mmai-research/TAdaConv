import torch


def tensor2cuda(data):
    """
    Put Tensor in iterable data into gpu.
    Args:
        data :(tensor or list or dict)
    """
    if type(data) == torch.Tensor:
        return data.cuda(non_blocking=True)
    elif type(data) == dict:
        keys = list(data.keys())
        for k in keys:
            data[k] = tensor2cuda(data[k])
    elif type(data) == list:
        for i in range(len(data)):
            data[i] = tensor2cuda(data[i])
    return data
