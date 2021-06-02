#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：fangpf
@Date    ：2021/6/2 15:56 
'''
import cv2
from PIL import Image
from torchvision import transforms

from model.mobilenetv2 import mobilenet_v2
from utils.utils import load_pretrained_weights
import torch.nn as nn
import numpy as np

model = mobilenet_v2(num_classes=10)
load_pretrained_weights(model, './weights/mobile_v2_last.pth')
model = model.cuda()
model.eval()

softmax = nn.Softmax()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.33708435, 0.42723662, 0.41629601], [0.2618102, 0.31948383, 0.33079577])
])

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


def demo():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.imread('data/imgs/test/img_83.jpg')
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        im = transform(im_pil)
        im = im.unsqueeze(0)
        im = im.cuda()
        out = model(im)
        y = softmax(out)
        # print(y)
        y = y.cpu().detach().numpy()
        idx = np.argmax(y, axis=1)[0]
        # print(y)
        conf = y[0][idx]
        # print(conf, idx)
        status = CLASS_STATUS[idx]
        print(status)

        cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0))
        cv2.putText(frame, str(conf), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0))
        cv2.imshow('status', frame)

        if cv2.waitKey(1) & ord('q') == 0xFF:
            break
        # fang[-1]

    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()
