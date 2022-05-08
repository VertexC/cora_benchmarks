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
V_SIZE = 128
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
                name = 'O', reduce_axis_ufs = [('k', ls[2])], width_uf_lists=[width_ufs_o])

    s = tvm.create_schedule([C.op])

    A_shared = s.cache_read(A, "shared", [C])
    A_local  = s.cache_read(A_shared, "local", [C])
    B_shared = s.cache_read(B, "shared", [C])
    B_local  = s.cache_read(B_shared, "local", [C])
    C_local = s.cache_write(C, "local")

    inputs = [[lens], [A, B, C]]

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

    thread_x = tvm.thread_axis("threadIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")

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
        runner=autotvm.LocalRunner(build_option=build_option, number=5, exec_func=exec_func))

    log_file = 'gemm_gpu_auto_{}.log'.format(args.dataset)
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=30,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(log_file)])


# apply history best from log file
with autotvm.apply_history_best(log_file):
    with tvm.target.create('cuda'):
        s, arg_bufs, size_fn = gemm()
        with tvm.build_config(prep_code_mode="with_prep_code",
                        fill_in_function_bodies=True,
                        hoist_loads=False,
                        disable_assert=False):
            func, i_bufs = tvm.build(s, arg_bufs, args.target, binds=None, substitutes=None)

time = exec_func(args.target, func, i_bufs, arg_bufs, size_fn)
print("final time:", time)