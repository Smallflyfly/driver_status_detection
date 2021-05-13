#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/07
"""
import argparse

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
import tensorboardX as tb
import torch.nn as nn
import torch
import numpy as np

from dataset import DriverStatusDataset
from utils.utils import build_optimizer, load_pretrained_weights, build_scheduler

parser = argparse.ArgumentParser(description="driver status detection")
parser.add_argument('--batch_size', default=16, type=int, help="train batch size")
parser.add_argument('--epoch', default=20, type=int, help="train epochs")
args = parser.parse_args()

batch_size = args.batch_size
max_epoch = args.epoch


def train():
    train_dataset = DriverStatusDataset('data/imgs', 'train')
    # test_dataset = DriverStatusDataset('data/imgs', 'test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # model = mobilenet_v3_small(num_classes=10)
    model = mobilenet_v2(num_classes=10)
    # load_pretrained_weights(model, 'weights/mobilenet_v3_small-047dcff4.pth')
    load_pretrained_weights(model, 'weights/mobilenet_v2-b0353104.pth')
    model = model.cuda()
    optimizer = build_optimizer(model, optim='adam', lr=0.0001)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=max_epoch)
    loss_func = nn.CrossEntropyLoss()
    softmax = nn.Softmax()
    writer = tb.SummaryWriter()

    for epoch in range(max_epoch):
        model.train()
        for index, data in enumerate(train_loader):
            im, label = data
            im = im.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()
            out = model(im)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                num_epoch = epoch * len(train_loader) + index
                print('Epoch: [{}/{}] [{}/{}]  loss = {:.6f}'.format(epoch + 1, max_epoch, index + 1, len(train_loader),
                                                                     loss))
                writer.add_scalar('loss', loss, num_epoch)
        scheduler.step()
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), 'weights/mobile_v2_net_relu_{}.pth'.format(epoch+1))
            # valid
            # model.eval()
            # total = len(test_loader)
            # sum_correct = 0
            # for data in test_loader:
            #     im, label = data
            #     im = im.cuda()
            #     y = model(im)
            #     y = softmax(y, dim=1)
            #     y = y.cpu().detach().numpy()
            #     idx = np.argmax(y, axis=1)[0]
            #     label = label.cpu().detach().numpy()[0]
            #     sum_correct += idx == label
            # accuracy = sum_correct / total
            # print('{} / {}  accuracy = {:.6f}'.format(epoch + 1, max_epoch, accuracy))
            # writer.add_scalar('accuracy', accuracy, epoch + 1)

    torch.save(model.state_dict(), 'weights/mobile_v2_relu_last.pth')
    writer.close()


if __name__ == '__main__':
    train()