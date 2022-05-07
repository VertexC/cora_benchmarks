from __future__ import print_function
# AutoTVM
import sys
# sys.path.insert(0,"/home/bowenc/dev/tvm/python")


import logging
import tvm
from tvm import te
import numpy as np
from tvm import autotvm
import numpy as np

from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw

import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils

import gc

parser = run_utils.get_cmd_parser()
args = parser.parse_args()


MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
print("get MAX_LEN:", MAX_LEN)
BATCH_SIZE = 32

# parameter
target = "cuda"
lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd, ld = Dim('bd'), Dim('ld')
def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens],
                                    lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw32 = len_ufw('s64', 64)

pad_sum = 64

@autotvm.template
def ewiseMul():
    ls =  {
        0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
        1: lufw1.get_uf(),
        2: lufw32.get_uf(),
    }

    loop_ufs=[ls[0], ls[1]]
    width_ufs=[ls[0], ls[2]]
    A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN), [bd, ld], loop_ufs, name='A', width_ufs=width_ufs)

    O = te.ragged_compute((BATCH_SIZE, MAX_LEN), [bd, ld], loop_ufs,
                lambda ds: 2 * A[ds[bd], ds[ld]], name = 'O', width_uf_lists=[width_ufs])

    s = tvm.create_schedule([O.op])

    ##### space definition begin #####
    thread_x = tvm.thread_axis("threadIdx.x")
    block_x = tvm.thread_axis("blockIdx.x")
    
    cfg = autotvm.get_config()

    x, y = s[O].op.axis
    cfg = autotvm.get_config()

    # TODO: add padding
    # print(tvm.lower(s, [A, O], target, simple_mode=True))

    f = s[O].fuse(x, y, padding=pad_sum)
    # import pdb; pdb.set_trace()
    cfg.define_split("tile_f", cfg.axis(run_utils.get_bound(x, ragged=True) * run_utils.get_bound(y, ragged=True)), num_outputs=2)
    # cfg.define_split("tile_y", y, num_outputs=2)
        
    # schedule according to config
    fo, fi = cfg["tile_f"].apply(s, O, f)
    # yo, yi = cfg["tile_y"].apply(s, O, y)

    # fo, fi = s[O].split(f, factor = 2)

    s[O].bind(fo, block_x)
    s[O].bind(fi, thread_x)

    def size_fn(l_inputs):
        lens = l_inputs[0] # batch [1, 2, 4, 5]
        fn = lufw32.get_fn(lens)
        return {
            A: run_utils.prefix_sum(len(lens), lambda b: (fn(b))),
            O: run_utils.prefix_sum(len(lens), lambda b: (fn(b)))
        }

    return s, [[lens], [A, O]], size_fn


task = autotvm.task.create(ewiseMul, args=(), target=target)

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


def exec_func(target, fadd, i_bufs, t_bufs, size_fn):
     
    host_i_inputs = []
    dev_i_inputs = []
    ctx = run_utils.get_ctx(target)
    cpu_ctx = run_utils.get_ctx("llvm")

    if len(i_bufs) == 2:
        host_i_inputs = [tvm.nd.array(run_utils.create_numpy_array(i, "int32"), cpu_ctx) for i in i_bufs[0]]
        dev_i_inputs = [tvm.nd.array(run_utils.create_numpy_array(i, "int32"), ctx) for i in i_bufs[1]]

    if size_fn is None:
        lw_args = {}
    batch_size = BATCH_SIZE
    batches = run_utils.get_nlp_batches(batch_size, args.max_batches, args.dataset, args.sort)
    batches = [sorted(batch, reverse=True) for batch in batches]
    batches = run_utils.append_padded_sum(batches, pad_sum)
    time = 0
    for i, batch in enumerate(batches):
        print("execute on batch ", i)
        t_inputs = [run_utils.create_tvm_array(i, "float32", ctx, rmap={}, lw_args=size_fn([batch]))
            for i in t_bufs[1]]
        l_inputs = [tvm.nd.array(batch, cpu_ctx)]
        inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
        time += run_utils.execute(target, fadd, inputs, ctx, False)
    gc.collect()
    return [time]

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.
build_option = {
                'prep_code_mode':'with_prep_code',
                'fill_in_function_bodies': True,
                'hoist_loads': False,
                'disable_assert': False,
            }
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(build_option=build_option, number=5, exec_func=exec_func))


# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.

tuner_method = "random"
if tuner_method == "random":
    tuner = autotvm.tuner.RandomTuner(task)
elif tuner_method == "xg":
    tuner = autotvm.tuner.XGBTuner(task)

log_file = 'ewise_dense_auto_{}.log'.format(tuner_method)
tuner.tune(n_trial=3,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file(log_file)])