import itertools
from itertools import chain
import torch
import torch.nn as nn
import utils
from utils import LRN

class AlexNet(nn.Module):

    def __init__(self, cudable, n_class):
        super(AlexNet, self).__init__()
        self.cudable = cudable
        self.n_class = n_class
        self.decay = 0.3
        self.s_centroid = torch.zeros(self.n_class, 256)
        self.t_centroid = torch.zeros(self.n_class, 256)
        if self.cudable:
            self.s_centroid = self.s_centroid.cuda()
            self.t_centroid = self.t_centroid.cuda()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            # nn.LocalResponseNorm(3, alpha=1e-5),
            LRN(local_size=5, alpha=1e-4, beta=0.75),
            nn.Conv2d(96, 256, 5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            LRN(local_size=5, alpha=1e-4, beta=0.75),
            # nn.LocalResponseNorm(3, alpha=1e-5),
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1)
        )
        self.init()

    def init(self):
        self.init_linear(self.fc8[0])
        self.init_linear(self.fc9[0], std=0.005)
        self.init_linear(self.D[0],D=True)
        self.init_linear(self.D[3],D=True)
        self.init_linear(self.D[6],D=True, std=0.3)
        self.CEloss, self.MSEloss, self.BCEloss = nn.CrossEntropyLoss(), nn.MSELoss(), nn.BCEWithLogitsLoss(reduction='mean')
        if self.cudable:
           self.CEloss, self.MSEloss, self.BCEloss = self.CEloss.cuda(), self.MSEloss.cuda(), self.BCEloss.cuda()

    def init_linear(self, m, std=0.01, D=False):
        # nn.init.normal_(m.weight.data, 0, std)
        # nn.init.xavier_normal_(m.weight)
        utils.truncated_normal_(m.weight.data, 0, std)
        if D:
            m.bias.data.fill_(0)
        else:
            m.bias.data.fill_(0.1)

    def forward(self, x, training=True):
        conv_out = self.features(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        dense_out = self.classifier(flattened)
        feature = self.fc8(dense_out)
        score = self.fc9(feature)
        pred = self.softmax(score)
        return feature, score, pred

    def forward_D(self, feature):
        logit = self.D(feature)
        return logit

    def closs(self, y_pred, y):
        C_loss = self.CEloss(y_pred, y)
        return C_loss

    def adloss(self, s_logits, t_logits, s_feature, t_feature, y_s, y_t):
        n, d = s_feature.shape

        # get labels
        s_labels, t_labels = y_s, torch.max(y_t, 1)[1]

        # image number in each class
        ones = torch.ones_like(s_labels, dtype=torch.float)
        zeros = torch.zeros(self.n_class)
        if self.cudable:
            zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones)

        # image number cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.n_class, d)
        if self.cudable:
            zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.n_class, 1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))

        # Moving Centroid
        decay = self.decay
        s_centroid = (1-decay) * self.s_centroid + decay * current_s_centroid
        t_centroid = (1-decay) * self.t_centroid + decay * current_t_centroid
        semantic_loss = self.MSEloss(s_centroid, t_centroid)
        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()

        # sigmoid binary cross entropy with reduce mean
        D_real_loss = self.BCEloss(t_logits, torch.ones_like(t_logits))
        D_fake_loss = self.BCEloss(s_logits, torch.zeros_like(s_logits))
        D_loss = (D_real_loss + D_fake_loss) * 0.1
        G_loss = -D_loss

        return G_loss, D_loss, semantic_loss

    # To some extent, can be replaced by weight_decay param in optimizer.
    def regloss(self):
        Dregloss = [torch.sum(layer.weight ** 2) / 2 for layer in self.D if type(layer) == nn.Linear]
        layers = chain(self.features, self.classifier, self.fc8, self.fc9)
        Gregloss = [torch.sum(layer.weight ** 2) / 2 for layer in layers if type(layer) == nn.Conv2d or type(layer) == nn.Linear]
        mean = lambda x:0.0005 * torch.mean(torch.stack(x))
        return mean(Dregloss), mean(Gregloss)


    def get_optimizer(self, init_lr, lr_mult, lr_mult_D):
        w_finetune, b_finetune, w_train, b_train, w_D, b_D = [], [], [], [], [], []

        finetune_layers = itertools.chain(self.features, self.classifier)
        train_layers = itertools.chain(self.fc8, self.fc9)
        for layer in finetune_layers:
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                w_finetune.append(layer.weight)
                b_finetune.append(layer.bias)
        for layer in train_layers:
            if type(layer) == nn.Linear:
                w_train.append(layer.weight)
                b_train.append(layer.bias)
        for layer in self.D:
            if type(layer) == nn.Linear:
                w_D.append(layer.weight)
                b_D.append(layer.bias)

        opt = torch.optim.SGD([{'params': w_finetune, 'lr': init_lr * lr_mult[0]},
                               {'params': b_finetune, 'lr': init_lr * lr_mult[1]},
                               {'params': w_train, 'lr': init_lr * lr_mult[2]},
                               {'params': b_train, 'lr': init_lr * lr_mult[3]}],
                              lr=init_lr,momentum=0.9)

        opt_D = torch.optim.SGD([{'params': w_D, 'lr': init_lr * lr_mult_D[0]},
                                 {'params': b_D, 'lr': init_lr * lr_mult_D[1]}],
                                lr=init_lr,momentum=0.9)

        return opt, opt_D
