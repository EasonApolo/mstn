from torch.utils import data
import cv2
import numpy as np

class Office(data.Dataset):

    def __init__(self, list, training=True):
        self.images = []
        self.labels = []
        self.multi_scale = [256, 257]
        self.output_size = [227, 227]
        self.training = training
        self.n_class = 31
        self.mean_color=[104.0069879317889,116.66876761696767,122.6789143406786]

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
            print 'Error: Image at {} not found.'.format(image_path)

        # set param
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            new_size = self.multi_scale[0]
        else:
            new_size = self.multi_scale[1]

        # resize
        img = img.astype(np.float32)
        img = cv2.resize(img, (new_size, new_size))

        # cropping
        if self.training:
            diff = new_size - self.output_size[0]
            offset_x = np.random.randint(0, diff, 1)[0]
            offset_y = np.random.randint(0, diff, 1)[0]
        else:
            offset_x = (img.shape[0] - self.output_size[0]) // 2
            offset_y = (img.shape[1] - self.output_size[1]) // 2
        img = img[offset_x:(offset_x+self.output_size[0]),
                  offset_y:(offset_y+self.output_size[1])]

        # substract mean
        img -= self.mean_color

        img = np.transpose(img, (2, 1, 0))
        
        # one hot label
        one_hot_label = np.zeros(self.n_class, dtype=np.long)
        one_hot_label[label] = 1

        return img, one_hot_label