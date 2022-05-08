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
V_SIZE = 128


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


    A_shared = s.cache_read(A, "shared", [C])
    A_local  = s.cache_read(A_shared, "local", [C])
    B_shared = s.cache_read(B, "shared", [C])
    B_local  = s.cache_read(B_shared, "local", [C])
    C_local = s.cache_write(C, "local")
    
    ##### space definition begin #####
    cfg = autotvm.get_config()

    b, x, y = s[C].op.axis

    cfg.define_knob("block_size", [1, 2, 4, 8, 16])
    cfg.define_knob("tx", [1, 2, 4, 8, 16])
    cfg.define_knob("ty", [1, 2, 4, 8, 16])

    # Save into the d2ltvm package.
    def split(stage, axis, factors):
        """Split an axis by a list of factors in a reverse order
        """
        axes = []
        for f in reversed(factors):
            axis, x = stage.split(axis, f)
            axes.append(x)
        return list(reversed(axes+[axis]))

    xb, xo, xi = split(s[C], x, (cfg["block_size"].val, cfg["tx"].val))
    yb, yo, yi = split(s[C], y, (cfg["block_size"].val, cfg["ty"].val))
    s[C].reorder(b, xb, yb, xo, yo, xi, yi)
    s[C].reorder(b, xb, yb, xo, yo, xi, yi)


    bxb = s[C].fuse(b, xb)

    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")

    s[C].bind(yb, block_x)
    s[C].bind(bxb, block_y)
    s[C].bind(yo, thread_x)
    s[C].bind(xo, thread_y)

    s[C_local].compute_at(s[C], yo)
    b, yi, xi = s[C_local].op.axis
    k, = s[C_local].op.reduce_axis

    cfg.define_split("tile_k", k, num_outputs=2)
    ko, ki = cfg["tile_k"].apply(s, C_local, k)
    s[C_local].reorder(ko, ki, yi, xi)

    def optimize_read_cache(shared, local, cfg_name):
        s[shared].compute_at(s[C_local], ko)
        s[local].compute_at(s[C_local], ki)
        b, y, x = s[shared].op.axis
        # Note that we must split into block_size parts to reuse
        # the previous axis threads
        yo, yi = s[shared].split(y, nparts=cfg["block_size"].val)
        xo, xi = s[shared].split(x, nparts=cfg["block_size"].val)
        s[shared].reorder(yo, xo, yi, xi)

        s[shared].bind(yo, thread_y)
        s[shared].bind(xo, thread_x)

    optimize_read_cache(A_shared, A_local, "cache_a")
    optimize_read_cache(B_shared, B_local, "cache_b")

    return s, [A, B, C]



# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

task = autotvm.task.create(
    "gemm", args=(BATCH_SIZE, E_SIZE, avg_seq_len, V_SIZE), target="cuda"
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
    callbacks=[autotvm.callback.log_to_file("gemm_gpu_vanilla_squad.log")],
)