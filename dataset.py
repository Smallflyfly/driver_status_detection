#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/07
"""
import os

import pandas
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DriverStatusDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        self.mode = mode
        self.data_root = data_root
        self.images = []
        self.labels = []
        self.image_label_map = {}
        self.cvs_file = 'data/lbls.csv'
        self.transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.33708435, 0.42723662, 0.41629601], [0.2618102,  0.31948383, 0.33079577])
        ])

        self.read_csv()
        self.prepare_data()

    def read_csv(self):
        csv = 'data/lbls.csv'
        dataframe = pandas.read_csv(csv)
        subjects = dataframe['subject']
        classnames = dataframe['classname']
        imgs = dataframe['img']
        for classname, img in zip(classnames, imgs):
            label = int(classname[-1])
            if img in self.image_label_map:
                print("rename of image!!")
                raise RuntimeError("rename of image!!")
            else:
                self.image_label_map[img] = label

    def prepare_data(self):
        if self.mode == 'train':
            folders = os.listdir(os.path.join(self.data_root, 'train'))
            for folder in folders:
                images = os.listdir(os.path.join(self.data_root, 'train', folder))
                for image in images:
                    label = self.image_label_map[image]
                    self.images.append(image)
                    self.labels.append(label)
        else:
            folders = os.listdir(os.path.join(self.data_root, 'test'))
            for folder in folders:
                images = os.listdir(os.path.join(self.data_root, 'test', folder))
                for image in images:
                    label = self.image_label_map[image]
                    self.images.append(image)
                    self.labels.append(label)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        im = Image.open(image)
        im = self.transforms(im)
        return im, label

    def __len__(self):
        return len(self.images)
