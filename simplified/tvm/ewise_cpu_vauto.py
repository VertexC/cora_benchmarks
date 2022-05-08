from __future__ import print_function
# AutoTVM
import sys
print(sys.path)
sys.path.insert(0,"/home/bowenc/dev/tvm/python")
print(sys.path)



import logging
import tvm
from tvm import te
import numpy as np
from tvm import autotvm
import numpy as np


@autotvm.template("ewiseAdd")
def ewiseAdd(M, N):
    # declare a matrix element-wise multiply
    A = te.placeholder((M, N), name="A")
    B = te.placeholder((M, N), name="B")
    C = te.compute((M, N), lambda i, j: A[i, j] * B[i, j], name="C")
    
    s = te.create_schedule([C.op])


    ##### space definition begin #####
    thread_x = te.thread_axis("threadIdx.x")
    block_x = te.thread_axis("blockIdx.x")
    
    ##### space definition begin #####
    cfg = autotvm.get_config()
    
    x, y = s[C].op.axis # why s[C]?

    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)
    
    # schedule according to config
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)

    fo  = s[C].fuse(xo, xi)
    fi  = s[C].fuse(yo, yi)

    s[C].bind(fo, block_x)
    s[C].bind(fi, thread_x)
    print(tvm.lower(s, [A, B], simple_mode=True))
    import pdb; pdb.set_trace()

    return s, [A, B, C]



# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# the last layer in resnet
M = 32
N = 32

# how it finds ewiseAdd???
task = autotvm.task.create(
    "ewiseAdd", args=(N, M), target="cuda"
)
print(task.config_space)

# Use local gpu, measure 10 times for every config to reduce variance
# The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
)

# Begin tuning, log records to file `conv2d.log`
# During tuning we will also try many invalid configs, so you are expected to
# see many error reports. As long as you can see non-zero GFLOPS, it is okay.
# import pdb; pdb.set_trace()
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=20,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("ewiseAdd.log")],
)