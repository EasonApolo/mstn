import cv2
import numpy as np
from torch.utils import data
import torchvision.transforms as tv


class Office(data.Dataset):

    def __init__(self, list, training=True):
        self.images = []
        self.labels = []
        self.multi_scale = [256, 257]
        self.output_size = [227, 227]
        self.training = training
        self.mean_color=[104.006,116.668,122.678]

        list_file = open(list)
        lines = list_file.readlines()
        for line in lines:
            fields = line.split()
            self.images.append(fields[0])
            self.labels.append(int(fields[1]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        img = cv2.imread(image_path)
        if type(img) == None:
            print('Error: Image at {} not found.'.format(image_path))

        if self.training and np.random.random() < 0.5:
            img = cv2.flip(img, 1)
        new_size = np.random.randint(self.multi_scale[0], self.multi_scale[1], 1)[0]

        img = cv2.resize(img, (new_size, new_size))
        img = img.astype(np.float32)

        # cropping
        if self.training:
            diff = new_size - self.output_size[0]
            offset_x = np.random.randint(0, diff, 1)[0]
            offset_y = np.random.randint(0, diff, 1)[0]
        else:
            offset_x = img.shape[0]//2 - self.output_size[0] // 2
            offset_y = img.shape[1]//2 - self.output_size[1] // 2

        img = img[offset_x:(offset_x+self.output_size[0]),
                  offset_y:(offset_y+self.output_size[1])]

        # substract mean
        img -= np.array(self.mean_color)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ToTensor transform cv2 HWC->CHW, only byteTensor will be div by 255.
        tensor = tv.ToTensor()
        img = tensor(img)
        # img = np.transpose(img, (2, 0, 1))

        return img, label