from __future__ import print_function
import os
import numpy as np
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils
from tvm import autotvm

parser = run_utils.get_cmd_parser()
args = parser.parse_args()
assert args.target == 'cuda'


MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
print("get MAX_LEN:", MAX_LEN)

@autotvm.template("test/ewiseAdd")
def ewiseAdd():
    # declare a matrix element-wise multiply
    BATCH_SIZE = te.var('bs')
    
    lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')
    bd, ld = Dim('bd'), Dim('ld')

    def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens],
                                    lambda lens: lambda b: utils.ceilmult(lens[b], pad))
    lufw1 = len_ufw('s1', 1)
    lufw32 = len_ufw('s64', 64)

    ls =  {
        0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
        1: lufw1.get_uf(),
        2: lufw32.get_uf(),
    }

    loop_ufs=[ls[0], ls[1]]
    width_ufs=[ls[0], ls[2]]
    A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN), [bd, ld], loop_ufs, name='A', width_ufs=width_ufs)

    O = te.ragged_compute((BATCH_SIZE, MAX_LEN), [bd, ld], loop_ufs,
                lambda ds: 2* A[ds[bd], ds[ld]], name = 'O', width_uf_lists=[width_ufs])

    s = tvm.create_schedule([O.op])
    
    ##### space definition begin #####
    cfg = autotvm.get_config()
    
    x, y = s[C].op.axis # why s[C]?


    # thread_x = tvm.thread_axis("threadIdx.x")
    # block_x = tvm.thread_axis("blockIdx.x")

    xo, xi = s[O].leaf_iter_vars
    f = s[O].fuse(xo, xi, padding = 64)

    cfg = autotvm.get_config()
    cfg.define_split("tile", f, num_outputs=2)

    # schedule according to config
    fo, fi = cfg["tile"].apply(s, C, f)

    # s[O].bind(fo, block_x)
    # s[O].bind(fi, thread_x)
    

    return s, [A, O]



# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

task = autotvm.task.create(
    "test/ewise", args=(), target="llvm"
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
 
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=20,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("ewise.log")],
)

inputs = [[lens], [BATCH_SIZE, A, O]]

def size_fn(l_inputs):
    lens = l_inputs[0]
    fn = lufw32.get_fn(lens)
    return {
        A: run_utils.prefix_sum(len(lens), lambda b: (fn(b))),
        O: run_utils.prefix_sum(len(lens), lambda b: (fn(b)))
    }
# prep_code_mode = 'with_prep_code'
# name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
# out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
#                                         run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE),
#                                         prep_code_mode=prep_code_mode, pad_sum=64)
