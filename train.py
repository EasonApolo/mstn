import torch
import torch.nn as nn
import numpy as np
import os
import math
import MyDataset
import MyModel
import utils
import argparse

def lr_schedule(opt, epoch, net='C'):
    lr = init_lr / pow(1 + 0.01 * epoch, 0.75)
    if net == 'D':
        lr_index = lr_mult_D
    elif net == 'C':
        lr_index = lr_mult
    elif net == 'A':
        lr_index = lr_mult_Alex
    for ind, param_group in enumerate(opt.param_groups):
        param_group['lr'] = lr * lr_index[ind]
    return lr


def adaptation_factor(x):
	if x>= 1.0:
		return 1.0
	den = 1.0 + math.exp(-10 * x)
	lamb = 2.0 / den - 1.0
	return lamb


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--resume', default=False, type=bool)
args = parser.parse_args()

s_ind = 0
t_ind = 1
init_lr = 1e-2
batch_size = 128
max_epoch = 10000
lr_mult = [0.1, 0.2, 1, 2]
lr_mult_D = [1, 2]
lr_mult_Alex = [0.3, 0.3]

# set
dataset_names = ['amazon', 'webcam', 'dslr']
s_name = dataset_names[s_ind]
t_name = dataset_names[t_ind]
n_class = 31
s_list_path = './data_list/' + s_name + '_list.txt'
t_list_path = './data_list/' + t_name + '_list.txt'
resume = args.resume

log = open('log/' + s_name + '_' + t_name + '_' + str(batch_size) + '.log', 'w')
checkpoint_path = 'checkpoint/' + s_name + '_' + t_name + '_' +str(batch_size) + '.pth'

gpu = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
cuda = torch.cuda.is_available()

print 'GPU: {}'.format(gpu)
print 'source: {}, target: {}, batch_size: {}, init_lr: {}'.format(s_name, t_name, batch_size, init_lr)
print 'lr_mult: {}, lr_mult_D: {}'.format(lr_mult, lr_mult_D)


# define DataLoader
s_loader = torch.utils.data.DataLoader(MyDataset.Office(s_list_path),
                                       batch_size=batch_size, shuffle=True, drop_last=True)
t_loader = torch.utils.data.DataLoader(MyDataset.Office(t_list_path),
                                       batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(MyDataset.Office(t_list_path, training=False),
                                         batch_size=1)
s_loader_len = len(s_loader)
t_loader_len = len(t_loader)

# define model
model = MyModel.AlexNet(cudable=cuda)

# define optimizer
opt, opt_D = model.get_optimizer(init_lr, lr_mult, lr_mult_D)

# resume or init
if resume:
    pretrain = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(pretrain['model'])
    opt.load_state_dict(pretrain['optimizer'])
    epoch = pretrain['epoch']
else:
    loaded_dict = utils.load_pretrain_npy()
    model.load_state_dict(loaded_dict, strict=False)
    epoch = 0

if cuda:
    model.cuda()
    print 'model cuda OK'


print '    =======    START TRAINING    =======    '
for epoch in range(epoch, max_epoch):

    model.train()

    if epoch % s_loader_len == 0:
        s_loader_epoch = iter(s_loader)
    if epoch % t_loader_len == 0:
        t_loader_epoch = iter(t_loader)
    xs, ys = s_loader_epoch.next()
    xt, yt = t_loader_epoch.next()

    lamb = adaptation_factor(epoch * 1.0 / max_epoch)
    current_lr = lr_schedule(opt, epoch, 'C')
    current_lr = lr_schedule(opt_D, epoch, 'D')

    opt.zero_grad()
    opt_D.zero_grad()

    if cuda:
        xs, ys = xs.cuda(), ys.cuda()
        xt, yt = xt.cuda(), yt.cuda()

    s_label = torch.max(ys, 1)[1]

    # forward
    s_feature, s_score, s_pred = model.forward(xs)
    t_feature, t_score, t_pred = model.forward(xt)
    s_logit = model.forward_D(s_feature)
    t_logit = model.forward_D(t_feature)

    # loss
    C_loss = model.closs(s_score, s_label)
    G_loss, D_loss, semantic_loss = model.adloss(s_logit, t_logit, s_feature, t_feature, ys, t_pred)
    Dregloss, Gregloss = model.regloss()
    F_loss = C_loss + lamb * G_loss + lamb * semantic_loss

    F_loss.backward(retain_graph=True)
    D_loss.backward()

    # optimize
    opt.step()
    opt_D.step()

    if epoch % 10 == 0:
        s_pred_label = torch.max(s_score, 1)[1]
        s_correct = torch.sum(torch.eq(s_pred_label, s_label).float())
        s_acc = torch.div(s_correct, ys.size(0))
        # log.write('epoch: {}, lr: {}, lambda: {}\n'.format(epoch, current_lr, lamb))
        # log.write('\tcorrect: {}, C_loss: {}, G_loss:{}, D_loss:{}, semantic_loss: {}, F_loss: {}\n'.format(s_correct.item(), C_loss.item(), G_loss.item(), D_loss.item(), semantic_loss.item(),  Floss.item()))
        print 'epoch: {}, lr: {}, lambda: {}'.format(epoch, current_lr, lamb)
        print 'correct: {}, C_loss: {}, G_loss:{}, D_loss:{}, semantic_loss: {}, F_loss: {}'.format(s_correct.item(), C_loss.item(), G_loss.item(), D_loss.item(), semantic_loss.item(),  F_loss.item())

    # validation
    if epoch % 50 == 0 and epoch != 0:
        print '    =======    START VALIDATION    =======    '
        model.eval()
        v_correct = 0
        v_sum = 0
        zeros = torch.zeros(n_class)
        zeros_classes = torch.zeros(n_class)
        if cuda:
            zeros = zeros.cuda()
            zeros_classes = zeros_classes.cuda()
        for ind2, (xv, yv) in enumerate(val_loader):
            if cuda:
                xv, yv = xv.cuda(), yv.cuda()
            v_feature, v_score, v_pred = model.forward(xv)
            v_pred_label = torch.max(v_score, 1)[1]
            v_label = torch.max(yv, 1)[1]
            v_equal = torch.eq(v_pred_label, v_label).float()
            zeros = zeros.scatter_add(0, v_label, v_equal)
            zeros_classes = zeros_classes.scatter_add(0, v_label, torch.ones_like(v_label, dtype=torch.float))
            v_correct += torch.sum(v_equal).item()
            v_sum += len(v_label)
        v_acc = v_correct / v_sum

        print 'validation: {}, {}'.format(v_correct, v_acc, zeros)
        print 'class: {}'.format(zeros.tolist())
        print 'class: {}'.format(zeros_classes.tolist())
        print 'source: {}, target: {}, batch_size: {}, init_lr: {}'.format(s_name, t_name, batch_size, init_lr)
        print 'lr_mult: {}, lr_mult_Alex: {}'.format(lr_mult, lr_mult_Alex)
        print '    =======    START TRAINING    =======    '

    # save model
    if epoch % 300 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': opt.state_dict()
        }, checkpoint_path)

    epoch += 1