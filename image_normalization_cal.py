#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/07
"""


import os
import numpy as np
import cv2

ROOT_PATH = 'data/imgs/train'


def cal_mean_std():
    mean, std = None, None
    folders = os.listdir(ROOT_PATH)
    for folder in folders:
        images = os.listdir(os.path.join(ROOT_PATH, folder))
        for image in images:
            im = cv2.imread(os.path.join(ROOT_PATH, folder, image))
            im = im[:, :, ::-1] / 255.
            # im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2])
            if mean is None and std is None:
                mean, std = cv2.meanStdDev(im)
            else:
                mean_, std_ = cv2.meanStdDev(im)
                mean_stack = np.stack((mean, mean_), axis=0)
                std_stack = np.stack((std, std_), axis=0)
                mean = np.mean(mean_stack, axis=0)
                std = np.mean(std_stack, axis=0)
    return mean.reshape((1, 3))[0], std.reshape((1, 3))[0]


if __name__ == '__main__':
    mean, std = cal_mean_std()
    print(mean, std)
    # [0.33708435 0.42723662 0.41629601] [0.2618102  0.31948383 0.33079577]