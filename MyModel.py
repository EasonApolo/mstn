import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.n_class = 31
        self.dropout = 0.5
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=5)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x, training=True):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        out = F.relu(pool1)
        return out
        