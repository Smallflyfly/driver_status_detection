#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:smallflyfly
@time: 2021/05/11
"""

import torch
from torchvision.models import mobilenet_v3_small

from utils.utils import load_pretrained_weights

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    model = mobilenet_v3_small(num_classes=10)
    load_pretrained_weights(model, './weights/last.pth')
    model.eval()
    model = model.cuda()
    output_onnx = "driver_status_detection.onnx"
    input_names = ['input']
    output_names = ['output']

    inputs = torch.randn(1, 3, 480, 640)
    inputs = inputs.cuda()
    onnx_out = torch.onnx.export(model, inputs, output_onnx, export_params=True, verbose=True,
                                 input_names=input_names, output_names=output_names, opset_version=11)