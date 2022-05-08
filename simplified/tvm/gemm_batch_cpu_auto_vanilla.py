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


BATCH_SIZE = 32
E_SIZE = 32
V_SIZE = 32


avg_seq_len = 190

@autotvm.template("gemm")
def gemm(bs, E_SIZE, L_SIZE, V_SIZE):
    
    k = te.reduce_axis((0, L_SIZE), name='k')
    A = te.placeholder((bs, E_SIZE, L_SIZE), name='A')
    B = te.placeholder((bs, L_SIZE, V_SIZE), name='B')
    C = te.compute((bs, E_SIZE, V_SIZE),
                    lambda b, x, y: te.sum(A[b, x, k] * B[b, k, y], axis=k),
                    name='C')


    s = te.create_schedule(C.op)
    CachedC = s.cache_write(C, 'local')
    # # Same as before, first tile by blocks, and then parallelize the
    # # computation of each block

    b, x, y = s[C].op.axis 
    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)

    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    s[C].reorder(b, xo, yo, xi, yi)

    xy = s[C].fuse(xo, yo)
    bxy = s[C].fuse(b, xy)

    s[C].parallel(bxy)
    # Use the write cache for the output of the xy axis, namely a block.
    s[CachedC].compute_at(s[C], bxy)
    # Same as before to optimize the computation of a block .
    b, xc, yc = s[CachedC].op.axis
    k = CachedC.op.reduce_axis[0]
    cfg.define_split("tile_k", k, num_outputs=2)
    ko, ki = cfg["tile_k"].apply(s, CachedC, k)

    s[CachedC].reorder(ko, xc, ki, yc)
    s[CachedC].unroll(ki)
    s[CachedC].vectorize(yc)

    return s, [A, B, C]



# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# how it finds ewiseAdd???
task = autotvm.task.create(
    "gemm", args=(BATCH_SIZE, E_SIZE, avg_seq_len, V_SIZE), target="llvm"
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
    n_trial=50,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("gemm_cpu_vanilla_squad.log")],
)