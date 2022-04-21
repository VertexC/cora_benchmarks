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

BATCH_SIZE = 32
E_SIZE = 2
V_SIZE = 2
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
print("get MAX_LEN:", MAX_LEN)
lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
ed = Dim('ed')
vd = Dim('vd')
ld = Dim('ld')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens],
                                   lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw32 = len_ufw('s64', 64)


@autotvm.template
def gemm():

    ls =  {
        0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
        1: lufw1.get_uf(),
        2: Uf.from_constant('ed', E_SIZE, 'l'),
        3: Uf.from_constant('vd', V_SIZE, 'l'),
        4: lufw32.get_uf(),
    }

    loop_ufs_a=[ls[0], ls[1], ls[2]]
    width_ufs_a=[ls[0], ls[4], ls[2]]

    loop_ufs_b=[ls[2], ls[3]]
    width_ufs_b=[ls[2], ls[3]]
    A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, E_SIZE), [bd, ld, ed], loop_ufs_a, name='A', width_ufs=width_ufs_a)
    B = te.ragged_placeholder((E_SIZE, V_SIZE), [ed, vd], loop_ufs_b, name='B', width_ufs=width_ufs_b)

    loop_ufs_o=[ls[0], ls[1], ls[3]]
    width_ufs_o=[ls[0], ls[4], ls[3]]

    O = te.ragged_compute((BATCH_SIZE, MAX_LEN, V_SIZE), [bd, ld, vd], loop_ufs_o,
                lambda ds, rds: tvm.sum(A[ds[bd], ds[ld], rds['k']] * B[rds['k'], ds[vd]], 
                    axis=rds['k'], dimensions = [ed]),
                name = 'O', reduce_axis_ufs = [('k', ls[2])], width_uf_lists=[width_ufs_o])

    s = tvm.create_schedule([O.op])

    inputs = [[lens], [A, B, O]]


    ##### space definition begin #####
    cfg = autotvm.get_config()


    # Blocking by loop tiling
    # import pdb; pdb.set_trace()
    x, y, z = s[O].op.axis
    k = s[O].op.reduce_axis[0]

    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_z", z, num_outputs=2)

    yo, yi = cfg["tile_y"].apply(s, O, y)
    zo, zi = cfg["tile_z"].apply(s, O, z)

    # Hoist reduction domain outside the blocking loop
    # s[O].reorder(x, yo, zo, k, yi, zi)

    def size_fn(l_inputs):
        lens = l_inputs[0] # batch [1, 2, 4, 5]
        fn = lufw32.get_fn(lens)
        return {
            A: run_utils.prefix_sum(len(lens), lambda b: (fn(b)*E_SIZE)),
            O: run_utils.prefix_sum(len(lens), lambda b: (fn(b)*V_SIZE))
        }
    return s, inputs, size_fn

def exec_func(target, fadd, i_bufs, t_bufs, size_fn):
    # import pdb; pdb.set_trace()
    run_function=run_utils.get_gemm_run_fn(BATCH_SIZE)
    avg_time = run_function(fadd, i_bufs, t_bufs[1], size_fn, args, pad_sum=None)
    return [avg_time]


task = autotvm.task.create(gemm, args=(), target=args.target)

build_option = {
                'prep_code_mode':'with_prep_code',
                'fill_in_function_bodies': True,
                'hoist_loads': False,
                'disable_assert': False,
            }

measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(build_option=build_option, number=5, exec_func=exec_func))

tuner_method = "random"
log_file = 'gemm_auto_{}.log'.format(tuner_method)
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=3,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file(log_file)])
