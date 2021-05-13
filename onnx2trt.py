#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/13
"""

import tensorrt as trt
import os

TRT_LOGGER = trt.Logger()
explicit_batch = 1
max_batch_size = 1


def get_engine(onnx_file, engine_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 20
        builder.max_batch_size = max_batch_size
        builder.fp16_mode = True
        if not os.path.exists(onnx_file):
            print("Onnx file {} not found!".format(onnx_file))
            return
        print("loading onnx file {}.".format(onnx_file))
        with open(onnx_file, 'rb') as onnx:
            parser.parse(onnx.read())

        last_layer = network.get_layer(network.num_layers - 1)
        network.mark_output(last_layer.get_output(0))

        engine = builder.build_cuda_engine(network)

        with open(engine_file, 'wb') as f:
            f.write(engine.serialize())

        return engine


if __name__ == '__main__':
    onnx_file = "driver_status_detection_mobile_v2_relu.onnx"
    engine_file = "driver_mobile_v2_relu.engine"
    get_engine(onnx_file, engine_file)
