#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:smallflyfly
@time: 2021/05/13
"""
import time

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from PIL import Image
import numpy as np
import cv2

engine_file = "driver_mobile_v2.engine"
TRT_LOGGER = trt.Logger()
im_mean = [0.33708435, 0.42723662, 0.41629601]
im_std = [0.2618102, 0.31948383, 0.33079577]


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # append the device buffer to device bindings
        bindings.append(int(device_mem))
        # append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def inference_v2(context, bindings, inputs, outputs, stream):
    # transfer input data  to the GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # run inference
    context.execute_async_v2(bindings, stream_handle=stream.handle)
    # transfer predictions from gpu to host
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # synchronize the stream
    stream.synchronize()

    return [out.host for out in outputs]


if __name__ == '__main__':
    with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
            runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        im = cv2.imread('test.jpg')
        # RGB BGR
        im = im[:, :, ::-1]
        im[:, :, 0] = (im[:, :, 0] / 255.0 - im_mean[0]) / im_std[0]
        im[:, :, 1] = (im[:, :, 1] / 255.0 - im_mean[1]) / im_std[1]
        im[:, :, 2] = (im[:, :, 2] / 255.0 - im_mean[2]) / im_std[2]
        im = im.reshape(1, 3, 480, 640)
        inputs[0].host = im
        print('inference...')
        start = time.time()
        out = inference_v2(context, bindings, inputs, outputs, stream)
        end = time.time()
        print('inference time: {}'.format(end - start))
        print(out)
        out = sofmax(out)