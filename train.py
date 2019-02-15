import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

import dataset
from model import AlexNet
import utils


def lr_schedule(opt, epoch, mult):
    lr = init_lr / pow(1 + 0.001 * epoch, 0.75)
    for ind, param_group in enumerate(opt.param_groups):
        param_group['lr'] = lr * mult[ind]
    return lr

def adaptation_factor(x):
	if x>= 1.0:
		return 1.0
	den = 1.0 + math.exp(-10 * x)
	lamb = 2.0 / den - 1.0
	return lamb

def output(mes):
    print(mes)
    # log.write(mes)


parser = argparse.ArgumentParser()
parser.add_argument('--s', default=0, type=int)
parser.add_argument('--t', default=1, type=int)
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--resume', default='', type=str)           # resume from pretrained bvlc_alexnet model
parser.add_argument('--da', default=1, type=int)                # 1 for doing domain adaptation
args = parser.parse_args()
resume = args.resume
da = args.da
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()

init_lr = 1e-2
batch_size = 100
max_epoch = 10000
lr_mult = [0.1, 0.2, 1, 2]
lr_mult_D = [1, 2]
dataset_names = ['amazon', 'webcam', 'dslr']
s_name = dataset_names[args.s]
t_name = dataset_names[args.t]
s_list_path = './data_list/' + s_name + '_list.txt'
t_list_path = './data_list/' + t_name + '_list.txt'
s_folder_path = '../../dataset/office/' + s_name + '/images'
t_folder_path = '../../dataset/office/' + t_name + '/images'
n_class = 31
log_name = '_'.join(str(a) for a in [s_name, t_name, str(batch_size), str(da), str(init_lr)])
log = open('log/' + log_name + '.log', 'w')
pretrain_path = 'checkpoint/' + resume + '.pth'
checkpoint_save_path = 'checkpoint/' + log_name + '.pth'
print('GPU: {}'.format(args.gpu))
print('source: {}, target: {}, batch_size: {}, init_lr: {}'.format(s_name, t_name, batch_size, init_lr))
print('lr_mult: {}, lr_mult_D: {}, resume: {}, da: {}'.format(lr_mult, lr_mult_D, resume, da))

s_loader = torch.utils.data.DataLoader(dataset.Office(s_list_path),
                                       batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
t_loader = torch.utils.data.DataLoader(dataset.Office(t_list_path),
                                       batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(dataset.Office(t_list_path, training=False),
                                         batch_size=1, num_workers=8)

s_loader_len, t_loader_len = len(s_loader), len(t_loader)

model = AlexNet(cudable=cuda, n_class=n_class)
if cuda:
    model.cuda()

opt, opt_D = model.get_optimizer(init_lr, lr_mult, lr_mult_D)

# resume or init
if not resume == '':
    pretrain = torch.load(pretrain_path)
    model.load_state_dict(pretrain['model'])
    opt.load_state_dict(pretrain['opt'])
    opt_D.load_state_dict(pretrain['opt_D'])
    epoch = pretrain['epoch'] # need change to 0 when DA
else:
    model.load_state_dict(utils.load_pth_model(), strict=False)
    # model.load_state_dict(utils.load_pretrain_npy(), strict=False)
    epoch = 0

output('    =======    START TRAINING    =======    ')
for epoch in range(epoch, 100000):
    model.train()
    lamb = adaptation_factor(epoch * 1.0 / max_epoch)
    current_lr, _ = lr_schedule(opt, epoch, lr_mult), lr_schedule(opt_D, epoch, lr_mult_D)

    if epoch % s_loader_len == 0:
        s_loader_epoch = iter(s_loader)
    if epoch % t_loader_len == 0:
        t_loader_epoch = iter(t_loader)
    xs, ys = s_loader_epoch.next()
    xt, yt = t_loader_epoch.next()
    if cuda:
        xs, ys, xt, yt = xs.cuda(), ys.cuda(), xt.cuda(), yt.cuda()

    # forward
    s_feature, s_score, s_pred = model.forward(xs)
    t_feature, t_score, t_pred = model.forward(xt)
    C_loss = model.closs(s_score, ys)
    if da:
        s_logit, t_logit = model.forward_D(s_feature), model.forward_D(t_feature)

        G_loss, D_loss, semantic_loss = model.adloss(s_logit, t_logit, s_feature, t_feature, ys, t_pred)
        Dregloss, Gregloss = model.regloss()
        F_loss = C_loss + Gregloss + lamb * G_loss + lamb * semantic_loss
        D_loss = D_loss + Dregloss

        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_D.step()
        opt.zero_grad()
        F_loss.backward(retain_graph=True)
        opt.step()
    else:
        opt.zero_grad()
        C_loss.backward()
        opt.step()

    if epoch % 10 == 0:
        s_pred_label = torch.max(s_score, 1)[1]
        s_correct = torch.sum(torch.eq(s_pred_label, ys).float())
        s_acc = torch.div(s_correct, ys.size(0))

        output('epoch: {}, lr: {}, lambda: {}'.format(epoch, current_lr, lamb))
        if da:
            output('correct: {}, C_loss: {}, G_loss:{}, D_loss:{}, Gregloss: {}, Dregloss: {}, semantic_loss: {}, F_loss: {}'.format(
                s_correct.item(), C_loss.item(), G_loss.item(), D_loss.item(),
                Gregloss.item(), Dregloss.item(), semantic_loss.item(), F_loss.item()))
        else:
            output('correct: {}, C_loss: {}'.format(s_correct.item(), C_loss.item()))


    # validation
    if epoch % 100 == 0 and epoch != 0:
        output('    =======    START VALIDATION    =======    ')
        model.eval()
        v_correct, v_sum = 0, 0
        zeros, zeros_classes = torch.zeros(n_class), torch.zeros(n_class)
        if cuda:
            zeros, zeros_classes = zeros.cuda(), zeros_classes.cuda()
        for ind2, (xv, yv) in enumerate(val_loader):
            if cuda:
                xv, yv = xv.cuda(), yv.cuda()
            v_feature, v_score, v_pred = model.forward(xv)
            v_pred_label = torch.max(v_score, 1)[1]
            v_equal = torch.eq(v_pred_label, yv).float()
            zeros = zeros.scatter_add(0, yv, v_equal)
            zeros_classes = zeros_classes.scatter_add(0, yv, torch.ones_like(yv, dtype=torch.float))
            v_correct += torch.sum(v_equal).item()
            v_sum += len(yv)
        v_acc = v_correct / v_sum
        output('validation: {}, {}'.format(v_correct, v_acc, zeros))
        output('class: {}'.format(zeros.tolist()))
        output('class: {}'.format(zeros_classes.tolist()))
        output('source: {}, target: {}, batch_size: {}, init_lr: {}'.format(s_name, t_name, batch_size, init_lr))
        output('lr_mult: {}, lr_mult_D: {}'.format(lr_mult, lr_mult_D))
        output('    =======    START TRAINING    =======    ')

    # save model
    if epoch % 1000 == 0 and epoch != 0:
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'opt_D': opt_D.state_dict()
        }, checkpoint_save_path)

    epoch += 1
