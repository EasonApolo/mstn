import torch
import os
import MyDataset
import MyModel
import itertools
import numpy as np
import torch.nn as nn
import utils

# params
gpu = 0
s_ind = 2
t_ind = 1
init_lr = 0.01
batch_size = 100
max_epoch = 10000


# set params
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
dataset_names = ['amazon', 'webcam', 'dslr']
s_name = dataset_names[s_ind]
t_name = dataset_names[t_ind]
n_class = 31
s_list_path = './data_list/' + s_name + '_list.txt'
t_list_path = './data_list/' + t_name + '_list.txt'
cuda = torch.cuda.is_available()

# define DataLoader
s_loader = torch.utils.data.DataLoader(MyDataset.Office(s_list_path),
                                       batch_size=batch_size, shuffle=True, drop_last=True)
t_loader = torch.utils.data.DataLoader(MyDataset.Office(t_list_path),
                                       batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(MyDataset.Office(t_list_path, training=False),
                                         batch_size=1)


# define model
model = MyModel.AlexNet()


# load pre-trained model
loaded_dict = utils.load_pretrain_npy()
model.load_state_dict(loaded_dict, strict=False)
model.cuda()

# define optimizer
opt = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

def adjust_learning_rate(opt, iter):
    lr = init_lr / pow(1 + 0.001 * iter, 0.75)
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr

# training
iter = 0
for epoch in range(0, max_epoch):
    for index, ([xs, ys], [xt, yt]) in enumerate(itertools.izip(s_loader, t_loader)):
        xs, ys = xs.cuda(), ys.cuda()
        raw = model.state_dict()
        opt.zero_grad()
        s_feature, s_score, s_pred = model.forward(xs)
        # t_feature, t_score, t_pred = model.forward(xt)
        # s_logit = model.forward_D(s_feature)
        # t_logit = model.forward_D(t_feature)

        s_pred_label = torch.max(s_pred, 1)[1]
        s_label = torch.max(ys, 1)[1]
        s_correct = torch.eq(s_pred_label, s_label).float()
        s_acc = torch.div(torch.sum(s_correct), ys.size(0))

        C_loss = model.closs(s_pred, s_label)
        C_loss.backward()

        # for param in model.parameters():
        #     if not param.grad is None:
        #         print param.grad.data.sum()

        # G_loss, D_loss, s_centroid, t_centroid = model.adloss(s_logit, t_logit, s_feature, t_feature, ys, t_pred)
        # G_loss.backward()


        current_lr = adjust_learning_rate(opt, iter)
        opt.step()

        iter += 1

        if item % 10 == 0:
            print torch.sum(s_correct).item(), s_acc.item()
            print C_loss.item()# , G_loss, D_loss
            print iter, current_lr

        # print G_loss, s_centroid
