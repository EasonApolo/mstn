import torch
import os
import MyDataset
import MyModel

gpu = 3
s_ind = 2
t_ind = 1

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

dataset_names = ['amazon', 'webcam', 'dslr']
s_name = dataset_names[s_ind]
t_name = dataset_names[t_ind]
n_class = 31
s_list_path = './data_list/' + s_name + '_list.txt'
t_list_path = './data_list/' + t_name + '_list.txt'

cuda = torch.cuda.is_available()

# data loader
s_loader = torch.utils.data.DataLoader(MyDataset.Office(s_list_path),
    batch_size=3, shuffle=True)
t_loader = torch.utils.data.DataLoader(MyDataset.Office(t_list_path),
    batch_size=3, shuffle=True)
val_loader = torch.utils.data.DataLoader(MyDataset.Office(t_list_path, training=False),
    batch_size=1)

model = MyModel.AlexNet()
# optimizer
# optim = torch.optim.

for i, (batch_x, batch_y) in enumerate(s_loader):
    out = model.forward(batch_x)
    print out.shape
    if i == 1:
        break