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


s = te.create_schedule([O.op])

prep_code_mode = "no_prep_code"
build_inputs = [[], [A, O]]
with tvm.build_config(prep_code_mode=prep_code_mode,
                          fill_in_function_bodies=True,
                          hoist_loads=False,
                          disable_assert=False):
    fadd, i_bufs = tvm.build(s, build_inputs, target, binds=None, substitutes=None)


    host_i_inputs = []
    dev_i_inputs = []
    ctx = tvm.cpu(0)
    t_inputs = [run_utils.create_tvm_array(i, "float32", ctx, rmap={}, lw_args={}) for i in build_inputs[1]]
    
    l_inputs = []
    inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
     
    time = run_utils.execute(target, fadd, inputs, ctx, False)
    print(time)