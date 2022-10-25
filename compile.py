#! /usr/bin/env python3

import pyopencl as cl

platform = cl.get_platforms()[0]
device = platform.get_devices(device_type=cl.device_type.GPU)
context = cl.Context(devices=device)
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
prg = open('dot.cl', 'r').read()
clprogram = cl.Program(context, prg)
out = clprogram.build()
print(out)

