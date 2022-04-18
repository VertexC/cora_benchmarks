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

# parameter
target = "llvm"
N, M = 1024, 1024

# declare a matrix element-wise multiply

@autotvm.template
def ewiseMul(M, N):
     
    A = te.placeholder((M, N), name="A")


    bd, ld = Dim('bd'), Dim('ld')

    ls =  {
        0: Uf.from_constant('bd', M, 'l'),
        1: Uf.from_constant('ld', N, 'l'),
    }

    loop_ufs=[ls[0], ls[1]]
    width_ufs=[ls[0], ls[1]]

    O = te.ragged_compute((M, N), [bd, ld], loop_ufs, lambda ds: A[ds[bd], ds[ld]] * 2, 
        name="O", width_uf_lists=[width_ufs])


    # xo, xi = s[O].leaf_iter_vars

    s = te.create_schedule([O.op])

    ##### space definition begin #####
    cfg = autotvm.get_config()

    x, y = s[O].op.axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)
        
    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, O, y)
    xo, xi = cfg["tile_x"].apply(s, O, x)

    s[O].reorder(yo, xo, yi, xi)

    return s, [[], [A, O]]


task = autotvm.task.create(ewiseMul, args=(N, M), target='llvm')

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))



def exec_func(target, fadd, i_bufs, args):
     
    host_i_inputs = []
    dev_i_inputs = []
    ctx = tvm.cpu(0)
    t_inputs = [run_utils.create_tvm_array(i, "float32", ctx, rmap={}, lw_args={}) for i in args[1]]
    
    l_inputs = []
    inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
    time = run_utils.execute(target, fadd, inputs, ctx, False)
    return [time]

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.
measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5, exec_func=exec_func))


# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.

tuner_method = "random"
# tuner = autotvm.tuner.RandomTuner(task)
tuner = autotvm.tuner.RandomTuner(task)

log_file = 'ewise_dense_auto_{}.log'.format(tuner_method)
tuner.tune(n_trial=3,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file(log_file)])

# apply history best from log file
with autotvm.apply_history_best(log_file):
    with tvm.target.create('llvm'):
        s, arg_bufs = ewiseMul(N, M)
        with tvm.build_config(prep_code_mode="no_prep_code",
                          fill_in_function_bodies=True,
                          hoist_loads=False,
                          disable_assert=False):
            func, i_bufs = tvm.build(s, arg_bufs, target, binds=None, substitutes=None)


time = exec_func(target, func, i_bufs, arg_bufs)
print("final time:", time)