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
E_SIZE = 32
V_SIZE = 32
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

    loop_ufs_b=[ls[0], ls[2], ls[3]]
    width_ufs_b=[ls[0], ls[2], ls[3]]
    A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, E_SIZE), [bd, ld, ed], loop_ufs_a, name='A', width_ufs=width_ufs_a)
    B = te.ragged_placeholder((BATCH_SIZE, E_SIZE, V_SIZE), [bd, ed, vd], loop_ufs_b, name='B', width_ufs=width_ufs_b)

    loop_ufs_o=[ls[0], ls[1], ls[3]]
    width_ufs_o=[ls[0], ls[4], ls[3]]

    C = te.ragged_compute((BATCH_SIZE, MAX_LEN, V_SIZE), [bd, ld, vd], loop_ufs_o,
                lambda ds, rds: tvm.sum(A[ds[bd], ds[ld], rds['k']] * B[ds[bd], rds['k'], ds[vd]], 
                    axis=rds['k'], dimensions = [ed]),
                name = 'C', reduce_axis_ufs = [('k', ls[2])], width_uf_lists=[width_ufs_o])

    s = tvm.create_schedule([C.op])

    CachedC = s.cache_write(C, 'local')

    inputs = [[lens], [A, B, C]]


    ##### space definition begin #####
    cfg = autotvm.get_config()


    # Blocking by loop tiling
    b, x, y = s[C].op.axis

    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)

    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)

    s[C].reorder(b, xo, yo, xi, yi)

    xy =  s[C].fuse(xo, yo)
    bxy = s[C].fuse(b, xy)
    
    s[C].parallel(bxy)

    s[CachedC].compute_at(s[C], bxy)
    b, xc, yc = s[CachedC].op.axis

    k = CachedC.op.reduce_axis[0]

    cfg.define_split("tile_k", k, num_outputs=2)
    ko, ki = cfg["tile_k"].apply(s, CachedC, k)

    s[CachedC].reorder(ko, xc, ki, yc)
    s[CachedC].unroll(ki)
    s[CachedC].vectorize(yc)


    def size_fn(l_inputs):
        lens = l_inputs[0] # batch [1, 2, 4, 5]
        fn = lufw32.get_fn(lens)
        return {
            A: run_utils.prefix_sum(len(lens), lambda b: (fn(b)*E_SIZE)),
            C: run_utils.prefix_sum(len(lens), lambda b: (fn(b)*V_SIZE))
        }
    return s, inputs, size_fn

def exec_func(target, fadd, i_bufs, t_bufs, size_fn):
    # import pdb; pdb.set_trace()
    run_function=run_utils.get_gemm_run_fn(BATCH_SIZE)
    avg_time = run_function(fadd, i_bufs, t_bufs[1], size_fn, args, pad_sum=None)
    return [avg_time]

log_file = args.log_file
if args.log_file == "":
    task = autotvm.task.create(gemm, args=(), target=args.target)

    build_option = {
                    'prep_code_mode':'with_prep_code',
                    'fill_in_function_bodies': True,
                    'hoist_loads': False,
                    'disable_assert': False,
                }

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(build_option=build_option, number=1, exec_func=exec_func))

    log_file = 'gemm_cpu_auto_{}.log'.format(args.dataset)
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=30,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(log_file)])


# apply history best from log file
with autotvm.apply_history_best(log_file):
    with tvm.target.create('llvm'):
        s, arg_bufs, size_fn = gemm()
        with tvm.build_config(prep_code_mode="with_prep_code",
                        fill_in_function_bodies=True,
                        hoist_loads=False,
                        disable_assert=False):
            func, i_bufs = tvm.build(s, arg_bufs, args.target, binds=None, substitutes=None)

time = exec_func(args.target, func, i_bufs, arg_bufs, size_fn)
print("final time:", time)



