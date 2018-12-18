# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LRN
import numpy as np
from itertools import chain

class AlexNet(nn.Module):

    def __init__(self, cudable):
        super(AlexNet, self).__init__()
        self.cudable = cudable
        self.n_class = 31
        self.decay = 0.3
        self.s_centroid = torch.zeros(256, self.n_class)
        self.t_centroid = torch.zeros(256, self.n_class)
        if self.cudable:
            self.s_centroid = self.s_centroid.cuda()
            self.t_centroid = self.t_centroid.cuda()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            LRN(local_size=1, alpha=1e-5, beta=0.75),

            nn.Conv2d(96, 256, 5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            LRN(local_size=1, alpha=1e-5, beta=0.75),

            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)
        )
        self.dense = nn.Sequential(
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096, 256)
        )
        self.fc9 = nn.Sequential(
            nn.Linear(256, self.n_class)
        )
        self.softmax = nn.Softmax(dim=0)
        self.D = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(),

            nn.Linear(1024, 1024),
            nn.Dropout(),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.mse = torch.nn.MSELoss()
        self.fc8.apply(self.init_weights)
        self.fc9.apply(self.init_weights)
        self.D[0].apply(self.init_weights)
        self.D[2].apply(self.init_weights)
        self.D[4].apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.1)
        
    def forward(self, x, training=True):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        dense_out = self.dense(conv_out)
        feature = self.fc8(dense_out)
        score = self.fc9(feature)
        pred = self.softmax(score)
        return feature, score, pred

    def forward_D(self, feature):
        logit = self.D(feature)
        return logit

    def closs(self, y_pred, y):
        CEloss = nn.CrossEntropyLoss()
        C_loss = CEloss(y_pred, y)
        return C_loss

    def adloss(self, s_logits, t_logits, s_feature, t_feature, y_s, y_t):
        n, d = s_feature.shape

        # get labels
        s_labels = torch.max(y_s, 1)[1]
        t_labels = torch.max(y_t, 1)[1]

        # n for each class
        ones = torch.ones_like(s_labels, dtype=torch.float)
        zeros = torch.zeros(self.n_class)
        if self.cudable:
            zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones)

        # class number cannot be 0, for calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        # class centroids
        zeros = torch.zeros(d, self.n_class)
        if self.cudable:
            zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(1, s_labels.repeat(d, 1), torch.transpose(s_feature, 0, 1))
        t_sum_feature = zeros.scatter_add(1, t_labels.repeat(d, 1), torch.transpose(t_feature, 0, 1))
        current_s_centroid = torch.div(s_sum_feature, s_n_classes)
        current_t_centroid = torch.div(t_sum_feature, t_n_classes)

        # Moving Centroid
        decay = self.decay
        s_centroid = (1-decay) * self.s_centroid + decay * current_s_centroid
        t_centroid = (1-decay) * self.t_centroid + decay * current_t_centroid

        # semantic Loss
        MSEloss = nn.MSELoss()
        semantic_loss = MSEloss(s_centroid, t_centroid)

        s_centroid = s_centroid.data
        t_centroid = t_centroid.data

        # sigmoid binary cross entropy with reduce mean
        BCEloss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        D_real_loss = BCEloss(t_logits, torch.ones_like(t_logits))
        D_fake_loss = BCEloss(s_logits, torch.zeros_like(s_logits))

        D_loss = D_real_loss + D_fake_loss
        G_loss = -D_loss * 0.1
        D_loss = D_loss * 0.1

        return G_loss, D_loss, semantic_loss

    def regloss(self):
        Dregloss = [self.mse(layer.weight, torch.zeros_like(layer.weight)) for layer in self.D if type(layer) == nn.Linear]
        layers = chain(self.conv, self.dense, self.fc8, self.fc9)
        Gregloss = [self.mse(layer.weight, torch.zeros_like(layer.weight)) for layer in layers if type(layer) == nn.Conv2d or type(layer) == nn.Linear]
        mean = lambda x:0.0005 * torch.mean(torch.stack(x))
        return mean(Dregloss), mean(Gregloss)