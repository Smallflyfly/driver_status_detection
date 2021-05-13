#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:smallflyfly
@time: 2021/05/08
"""
import cv2
import torch
from PIL import Image
# from torchvision.models import mobilenet_v3_small, mobilenet_v2
from model.mobilenetv2 import mobilenet_v2
from utils.utils import load_pretrained_weights
import torch.nn as nn
import torchvision.transforms.transforms as transforms
import numpy as np

CLASS_STATUS = {
    0: 'normal driving',
    1: 'texting - right',
    2: 'talking on the phone - right',
    3: 'texting - left',
    4: 'talking on the phone - left',
    5: 'operating the radio',
    6: 'drinking',
    7: 'reaching behind',
    8: 'hair and makeup',
    9: 'talking to passenger'
}


def demo(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.33708435, 0.42723662, 0.41629601], [0.2618102, 0.31948383, 0.33079577])
    ])
    model = mobilenet_v2(num_classes=10)
    # load_pretrained_weights(model, './weights/net_20.pth')
    load_pretrained_weights(model, './weights/mobile_v2_net_relu_16.pth')
    model = model.cuda()
    model.eval()
    # print(model)
    softmax = nn.Softmax()
    im = Image.open(image)
    im = transform(im)
    im = im.unsqueeze(0)
    im = im.cuda()
    out = model(im)
    y = softmax(out)
    y = y.cpu().detach().numpy()
    idx = np.argmax(y, axis=1)[0]
    # print(y)
    conf = y[0][idx]
    print(conf, idx)
    status = CLASS_STATUS[idx]
    im_cv = cv2.imread(image)
    cv2.putText(im_cv, status, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0))
    cv2.putText(im_cv, str(conf), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0))
    cv2.imshow('status', im_cv)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # image = 'data/imgs/test/img_1.jpg'
    image = 'test.jpg'
    demo(image)
