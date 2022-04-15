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

    return s, [A, O]


task = autotvm.task.create(ewiseMul, args=(N, M), target='llvm')
print(task.config_space)