# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LRN
import numpy as np

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.n_class = 31
        self.decay = 0.3
        self.s_centroid = torch.zeros(256, self.n_class)
        self.t_centroid = torch.zeros(256, self.n_class)
        self.lr = 0.01
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
        self.fc8 = (
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
        self.fc8.apply(self.init_weights)
        self.fc9.apply(self.init_weights)

        # self.opt_conv = torch.optim.SGD(self.conv.parameters(), lr = self.lr * 1, momentum=0.9)
        # self.opt_dense = torch.optim.SGD(self.dense.parameters(), lr = self.lr * 2, momentum=0.9)
        # self.opt_score = torch.optim.SGD(self.score.parameters(), lr = self.lr * 1, momentum=0.9)
        # self.opt_D = torch.optim.SGD(self.D.parameters(), lr = self.lr * 2, momentum=0.9)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        
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
        self.C_loss = CEloss(y_pred, y)
        return self.C_loss

    def adloss(self, s_logits, t_logits, s_feature, t_feature, y_s, y_t):
        n, d = s_feature.shape

        # get labels
        s_labels = torch.max(y_s, 1)[1]
        t_labels = torch.max(y_t, 1)[1]

        # 每类样本数
        ones = torch.ones_like(s_labels, dtype=torch.float)
        s_n_classes = torch.zeros(self.n_class).scatter_add(0, s_labels, ones)
        t_n_classes = torch.zeros(self.n_class).scatter_add(0, t_labels, ones)

        # 类别数不能为0，待会算类中心要除以类别数
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        # 每类中心
        s_sum_feature = torch.zeros(d, self.n_class).scatter_add(1, s_labels.repeat(d, 1), torch.transpose(s_feature, 0, 1))
        t_sum_feature = torch.zeros(d, self.n_class).scatter_add(1, t_labels.repeat(d, 1), torch.transpose(t_feature, 0, 1))
        current_s_centroid = torch.div(s_sum_feature, s_n_classes)
        current_t_centroid = torch.div(t_sum_feature, t_n_classes)

        # Moving Centroid
        decay = self.decay
        s_centroid = (1-decay) * self.s_centroid + decay * current_s_centroid
        t_centroid = (1-decay) * self.t_centroid + decay * current_t_centroid

        # semantic Loss
        self.semantic_loss = torch.mean(s_centroid - t_centroid).pow(2)

        # sigmoid binary cross entropy with reduce mean
        BCEloss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        D_real_loss = BCEloss(t_logits, torch.ones_like(t_logits))
        D_fake_loss = BCEloss(s_logits, torch.zeros_like(s_logits))

        D_loss = D_real_loss + D_fake_loss
        G_loss = -D_loss * 0.1
        self.D_loss = D_loss * 0.1
        self.G_loss = G_loss

        return G_loss, D_loss, s_centroid, t_centroid

    def adv_optimizer(self):
        return

    def optimizer(self):
        F_loss = self.C_loss + self.semantic_loss + self.G_loss + self.D_loss
        F_loss.backward()

        self.opt_conv.zero_grad()
        self.opt_dense.zero_grad()
        # self.opt_D.zero_grad()
        self.opt_score.zero_grad()

        self.opt_conv.step()
        self.opt_dense.step()
        # self.opt_D.step()
        self.opt_score.step()
