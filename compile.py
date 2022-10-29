#! /usr/bin/env python3

import pyopencl as cl
import numpy as np

platform = cl.get_platforms()[0]
device = platform.get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=device)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
prg = open('dot.cl', 'r').read()
clprogram = cl.Program(ctx, prg)
pg = clprogram.build()

input_shape = (10,1)
seed = 42
g = np.random.default_rng(seed=seed)
input1 = g.integers(0, 255, size=(input_shape)).astype(np.float32)
input2 = g.integers(0, 255, size=(input_shape)).astype(np.float32)
expected = input1.transpose() @ input2

output = np.empty_like(expected)

mf = cl.mem_flags
g1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input1)
g2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input2)
o1 = cl.Buffer(ctx, mf.WRITE_ONLY, expected.nbytes*2)

reduce0 = pg.reduce_0
reduce0(queue, 
        output.shape, # global_size
        None,         # local_size
        o1,           # *args
        g1,
        g2)
    
cl.enqueue_copy(queue, output, o1)
print(output)
print(np.linalg.norm(output-expected))
assert np.allclose(output, expected)