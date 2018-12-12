import torch
import os
import MyDataset

gpu = 3
s_ind = 2
t_ind = 1

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

dataset_names = ['amazon', 'webcam', 'dslr']
s_name = dataset_names[s_ind]
t_name = dataset_names[t_ind]
n_class = 31
s_list_path = '../../../Moving-Semantic-Transfer-Network/office/data/' + s_name + '_list.txt'
t_list_path = '../../../Moving-Semantic-Transfer-Network/office/data/' + t_name + '_list.txt'

cuda = torch.cuda.is_available()

s_loader = torch.utils.data.DataLoader(MyDataset.Office(s_list_path), batch_size=3)

for i, data in enumerate(s_loader):
    print 'a'