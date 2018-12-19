import torch
import os
import MyDataset
import MyModel
import itertools
import numpy as np
import torch.nn as nn
import utils
import math

def adjust_learning_rate(opt, iter):
    lr = init_lr / pow(1 + 0.001 * iter, 0.75)
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr

def adaptation_factor(x):
	if x>=1.0:
		return 1.0
	den=1.0+math.exp(-10*x)
	lamb=2.0/den-1.0
	return lamb

# params
s_ind = 1
t_ind = 2
init_lr = 0.01
batch_size = 100
max_epoch = 10000

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
cuda = torch.cuda.is_available()

# set params
dataset_names = ['amazon', 'webcam', 'dslr']
s_name = dataset_names[s_ind]
t_name = dataset_names[t_ind]
n_class = 31
s_list_path = './data_list/' + s_name + '_list.txt'
t_list_path = './data_list/' + t_name + '_list.txt'
log = open('log/' + s_name + '_' + t_name + '_' + str(batch_size) + '.log', 'w')
print 'source: {}, target: {}, batch_size: {}'.format(s_name, t_name, batch_size)

# define DataLoader
s_loader = torch.utils.data.DataLoader(MyDataset.Office(s_list_path),
                                       batch_size=batch_size, shuffle=True, drop_last=True)
t_loader = torch.utils.data.DataLoader(MyDataset.Office(t_list_path),
                                       batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(MyDataset.Office(t_list_path, training=False),
                                         batch_size=batch_size)


# define model
model = MyModel.AlexNet(cudable=cuda)

print 'start loading model...'
# load pre-trained model
loaded_dict = utils.load_pretrain_npy()
model.load_state_dict(loaded_dict, strict=False)
print 'model loading OK'
if cuda:
    model.cuda()
print 'model.cuda OK'

# define optimizer
conv_params = list(map(id, model.conv.parameters()))
dense_params = list(map(id, model.dense.parameters()))

opt = torch.optim.SGD([{'params': model.conv.parameters(), 'lr': init_lr*0.1},
                       {'params': model.dense.parameters(), 'lr': init_lr * 0.1},
                       {'params': model.fc8.parameters(), 'lr': init_lr * 1},
                       {'params': model.fc9.parameters(), 'lr': init_lr * 2}
                       ], lr=init_lr, momentum=0.9)

opt_D = torch.optim.SGD(params=model.D.parameters(), lr=init_lr, momentum=0.9)
opt_D = torch.optim.SGD(params=model.D.parameters(), lr=init_lr, momentum=0.9)

print 'training start'
# training
iter = 0
for epoch in range(0, max_epoch):
    for index, ([xs, ys], [xt, yt]) in enumerate(itertools.izip(s_loader, t_loader)):
        # set mode and params for this iter
        iter += 1
        model.train()
        lamb = adaptation_factor(iter*1.0/max_epoch)
        current_lr = adjust_learning_rate(opt, iter)

        opt.zero_grad()
        opt_D.zero_grad()

        if cuda:
            xs, ys = xs.cuda(), ys.cuda()
            xt, yt = xt.cuda(), yt.cuda()

        # forward C
        s_feature, s_score, s_pred = model.forward(xs)
        t_feature, t_score, t_pred = model.forward(xt)
        s_logit = model.forward_D(s_feature)
        t_logit = model.forward_D(t_feature)

        # compute src accuracy
        s_pred_label = torch.max(s_pred, 1)[1]
        s_label = torch.max(ys, 1)[1]
        s_correct = torch.sum(torch.eq(s_pred_label, s_label).float())
        s_acc = torch.div(s_correct, ys.size(0))

        # loss
        C_loss = model.closs(s_pred, s_label)
        G_loss, D_loss, semantic_loss = model.adloss(s_logit, t_logit, s_feature, t_feature, ys, t_pred)
        Dregloss, Gregloss = model.regloss()
        Floss = C_loss + Gregloss + lamb * G_loss + lamb * semantic_loss

        # backward
        Floss.backward(retain_graph=True)
        Dregloss.backward()

        # optimize
        opt.step()
        opt_D.step()

        # validation
        if iter % 50 == 0:
            print 'start validation'
            model.eval()
            v_correct = 0
            v_sum = 0
            for ind2, (xv, yv) in enumerate(val_loader):
                if cuda:
                    xv, yv = xv.cuda(), yv.cuda()
                v_feature, v_score, v_pred = model.forward(xv)
                v_pred_label = torch.max(v_pred, 1)[1]
                v_label = torch.max(yv, 1)[1]
                v_correct += torch.sum(torch.eq(v_pred_label, v_label).float())
                v_sum += len(v_label)
            v_acc = torch.div(v_correct, v_sum)
            log.write('validation: {}, {}\n'.format(v_correct.item(), v_acc.item()))
            print 'validation: {}, {}'.format(v_correct.item(), v_acc.item())

        # print and log
        if iter % 1 == 0:
            log.write('iter: {}, lr: {}, lambda: {}\n'.format(iter, current_lr, lamb))
            log.write('\tcorrect: {}, C_loss: {}, G_loss:{}, D_loss:{}, semantic_loss: {}, Dregloss: {}, Gregloss: {}, F_loss: {}\n'.format(s_correct.item(), C_loss.item(), G_loss.item(), D_loss.item(), semantic_loss.item(), Dregloss.item(), Gregloss.item(),  Floss.item()))
            print 'iter: {}, lr: {}, lambda: {}'.format(iter, current_lr, lamb)
            print 'correct: {}, C_loss: {}, G_loss:{}, D_loss:{}, semantic_loss: {}, Dregloss: {}, Gregloss: {}, F_loss: {}'.format(s_correct.item(), C_loss.item(), G_loss.item(), D_loss.item(), semantic_loss.item(), Dregloss.item(), Gregloss.item(),  Floss.item())

        # save model
        if iter % 2000 == 0:
            torch.save({
                'epoch': epoch,
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'opt_D_state_dict': opt_D.state_dict()
            }, PATH)