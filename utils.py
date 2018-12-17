import torch
import torch.nn as nn
import collections
import numpy as np

# Local Response Normalization implementation from https://github.com/pytorch/pytorch/issues/653
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


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

        # reverse all dimension for matching, shape==2 is fc, shape==4 is conv
        if len(weight.shape) == 2:
            weight = np.transpose(weight, (1, 0))
        elif len(weight.shape) == 4:
            weight = np.transpose(weight, (3, 2, 1, 0))

        # add keys and data
        new_dict[newkey + '.weight'] = torch.Tensor(weight)
        new_dict[newkey + '.bias'] = torch.Tensor(old_dict[key][1])

    return new_dict