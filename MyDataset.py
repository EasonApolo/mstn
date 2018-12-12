import torch
from torch.utils import data
import cv2
from numpy.random import random

class Office(data.Dataset):

    def __init__(self, list):
        self.images = []
        self.labels = []

        list_file = open(list)
        lines = list_file.readlines()
        for line in lines:
            fields = line.split()
            self.images.append(fields[0])
            self.labels.append(fields[1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]

        img = cv2.imread(image_path)
        if random() < 0.5:
            print 'into random'

        print image_path
        print img.shape
        print label

        return img, label