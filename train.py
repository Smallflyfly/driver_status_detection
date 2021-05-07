#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/07
"""
import argparse

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

from dataset import DriverStatusDataset
from utils.utils import build_optimizer, load_pretrained_weights, build_scheduler

parser = argparse.ArgumentParser(description="driver status detection")
parser.add_argument('--batch_size', default=16, type=int, help="train batch size")
parser.add_argument('--epoch', default=100, type=int, help="train epochs")
args = parser.parse_args()

batch_size = args.batch_size
max_epoch = args.epoch


def train():
    train_dataset = DriverStatusDataset('data/imgs', 'train')
    test_dataset = DriverStatusDataset('data/imgs', 'test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = mobilenet_v3_small(num_classes=10)
    load_pretrained_weights('weights/mobilenet_v3_small-047dcff4.pth')
    model = model.cuda()
    optimizer = build_optimizer(model, optim='adam', lr=0.0001)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine')


if __name__ == '__main__':
    train()