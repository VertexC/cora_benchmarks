import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils

parser = run_utils.get_cmd_parser(no_options=True)
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--m', dest='m', default=1024, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')
args = parser.parse_args()

M = args.m
md = Dim('md')
nd = Dim('nd')
kd = Dim('kd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: utils.ceilmult(m + 1, pad))
lufw = len_ufw('s2k', 32)

luf = lufw.get_uf()
ls =  {
    0: Uf.from_constant('md', M, 'l'),
    1: Uf.from_constant('nd', M, 'l'),
    2: Uf.from_constant('kd', M, 'l'),
}

loop_ufs=[ls[0], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((M, M), [md, kd], loop_ufs, name='A', width_ufs=None)

B = te.placeholder((M, M), name='B')

loop_ufs=[ls[0], ls[1]]
O = te.ragged_compute((M, M), [md, nd], loop_ufs,
                      # lambda ds, rds: tvm.sum(tvm.tir.Cast('int32', rds['k'] > (ds[md] + 1)) *
                                              # A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                              # axis=rds['k'], dimensions = [kd]),
                      lambda ds, rds: tvm.sum(tvm.if_then_else(rds['k'] > (ds[md] + 1),
                                                               A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                                               0),
                                              axis=rds['k'], dimensions = [kd]),
                      # lambda ds, rds: tvm.sum(A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                              # axis=rds['k'], dimensions = [kd]),
                      name = 'O', reduce_axis_ufs = [('k', luf)], width_uf_lists=None)

s = tvm.create_schedule([O.op])

if False:
    O_local, = s.cache_write([O], "local", storage_layout_mode='loop_layout')

    l, o, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    loi, li = s[O_local].split(l, factor=2)

    ooi, oi = s[O_local].split(o, factor=2)

    koi, ki = s[O_local].split(k, factor=4)
    koo, koi = s[O_local].split(koi, factor=2)

    s[O_local].reorder(koo, koi, loi, ooi, ki, li, oi)

    if not args.debug_code:
        s[O_local].unroll(koi)
        s[O_local].unroll(loi)
        s[O_local].unroll(ooi)
        s[O_local].unroll(ki)
        s[O_local].unroll(li)
        s[O_local].unroll(oi)

    O_l, O_o, O_k = tuple(O.op.axis) + tuple(O.op.reduce_axis)

    O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
    O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=2)
    O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

    O_o_o_i, O_o_i = s[O].split(O_o, factor=4)
    O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
    O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=1)
    s[O].reorder(O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)

    A_shared = s.cache_read(A, "shared", [O_local])
    A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
    s[A_shared].compute_at(s[O_local], koo)

    B_shared = s.cache_read(B, "shared", [O_local], vanilla=True)
    B_shared_ax0, B_shared_ax1 = tuple(B_shared.op.axis)
    s[B_shared].compute_at(s[O_local], koo)

    s[O].bind(O_l_o_o_o, te.thread_axis("blockIdx.y"))
    s[O].bind(O_o_o_o_o, te.thread_axis("blockIdx.x"))
    O_l_o_o_i_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
    s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
    O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
    s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
    s[O_local].compute_at(s[O], O_l_o_i_o_o_i_fused)

    A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
    A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
    if not args.debug_functions: s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
    A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
    s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
    s[A_shared].mark_no_bounds_check()

    B_shared_ax0_ax1_fused = s[B_shared].fuse(B_shared_ax0, B_shared_ax1)
    B_shared_ax0_ax1_fused_o, B_shared_ax0_ax1_fused_i = s[B_shared].split(B_shared_ax0_ax1_fused, factor=4)
    if not args.debug_functions: s[B_shared].vectorize(B_shared_ax0_ax1_fused_i)
    B_shared_ax0_ax1_fused_o_o, B_shared_ax0_ax1_fused_o_i = s[B_shared].split(B_shared_ax0_ax1_fused_o, factor=32)
    s[B_shared].bind(B_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

    s[O].mark_no_bounds_check()
    s[O_local].mark_no_bounds_check()
else:
    S, = s.cache_write([O], "local", storage_layout_mode='loop_layout')

    S_l, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)
    S_l_o_i, S_l_i = s[S].split(S_l, factor=4)
    S_l_o_o_i, S_l_o_i = s[S].split(S_l_o_i, factor=8)

    S_k_o_o, S_k_o_i = s[S].split(S_k, factor=4)
    s[S].reorder(S_l_o_o_i, S_k_o_o, S_k_o_i, S_l_o_i, S_o, S_l_i)
    # s[S].peel(S_k_o_o)
    s[S].unroll(S_l_i)

    O_l, O_o, O_k = tuple(O.op.axis) + tuple(O.op.reduce_axis)
    O_l_o_i, O_l_i = s[O].split(O_l, factor=32)

    O_o_o_o_i, O_o_o_i = s[O].split(O_o, factor=64)
    O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=2)

    s[O].reorder(O_l_o_i, O_o_o_o_o, O_o_o_o_i, O_o_o_i, O_l_i)
    s[S].compute_at(s[O], O_o_o_i)

    A_shared = s.cache_read(A, "shared", [S])
    A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
    s[A_shared].compute_at(s[S], S_k_o_o)

    B_shared = s.cache_read(B, "shared", [S], vanilla=True)
    B_shared_ax0, B_shared_ax1 = tuple(B_shared.op.axis)
    s[B_shared].compute_at(s[S], S_k_o_o)

    O_l_o_i_o_o_o_o_fused = s[O].fuse(O_l_o_i, O_o_o_o_o)
    s[O].bind(O_l_o_i_o_o_o_o_fused, te.thread_axis("blockIdx.x"))
    s[O].bind(O_o_o_o_i, te.thread_axis("vthread"))
    s[O].bind(O_o_o_i, te.thread_axis("threadIdx.x"))

    A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
    A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
    s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
    A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=64)
    s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

    B_shared_ax0_ax1_fused = s[B_shared].fuse(B_shared_ax0, B_shared_ax1)
    B_shared_ax0_ax1_fused_o, B_shared_ax0_ax1_fused_i = s[B_shared].split(B_shared_ax0_ax1_fused, factor=4)
    s[B_shared].vectorize(B_shared_ax0_ax1_fused_i)
    B_shared_ax0_ax1_fused_o_o, B_shared_ax0_ax1_fused_o_i = s[B_shared].split(B_shared_ax0_ax1_fused_o, factor=64)
    s[B_shared].bind(B_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

    # s[S].pragma(S_l_o_o_o_o, "auto_unroll_max_step", 512)
    # s[S].pragma(S_l_o_o_o_o, "unroll_explicit", True)

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

inputs = [[], [A, B, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, run_function=run_utils.run_trmm,
                               prep_code_mode='no_prep_code')

# _, A, B, O  = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     rounded64 = utils.ceilmult(length, 64)
#     this_extent = rounded64 * NUM_HEADS * HEAD_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]))
#     ctr += this_extent
