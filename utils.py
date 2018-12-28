import torch
import torch.nn as nn
import collections
import numpy as np


# load pre-trained alexnet model, and return a new Dictionary with matched keys
def load_pretrain_npy():
    old_dict = np.load('bvlc_alexnet.npy', encoding='bytes').item()
    new_dict = collections.OrderedDict()
    for key in old_dict:
        if key == 'conv1':
            newkey = 'conv.0'
        elif key == 'conv2':
            newkey = 'conv.4'
        elif key == 'conv3':
            newkey = 'conv.8'
        elif key == 'conv4':
            newkey = 'conv.10'
        elif key == 'conv5':
            newkey = 'conv.12'
        elif key == 'fc6':
            newkey = 'dense.0'
        elif key == 'fc7':
            newkey = 'dense.3'
        else:
            continue
        weight = old_dict[key][0]
        bias = old_dict[key][1]

        # reverse all dimension for matching, shape==2 is fc, shape==4 is conv
        if len(weight.shape) == 2:
            weight = np.transpose(weight, (1, 0))
        elif len(weight.shape) == 4:
            weight = np.transpose(weight, (3, 2, 0, 1))

        # add keys and data
        t = torch.from_numpy(weight)
        new_dict[newkey + '.weight'] = t
        new_dict[newkey + '.bias'] = torch.Tensor(bias)

    return new_dict


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)