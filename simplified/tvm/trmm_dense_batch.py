import numpy as np
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
parser.add_argument('--n', dest='n', default=128, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')
args = parser.parse_args()
assert args.target == "cuda"

M = args.m
N = args.n
md = Dim('md')
nd = Dim('nd')
kd = Dim('kd')

ls =  {
    0: Uf.from_constant('md', M, 'l'),
    1: Uf.from_constant('nd', N, 'l'),
    2: Uf.from_constant('kd', M, 'l'),
}

loop_ufs=[ls[0], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((M, M), [md, kd], loop_ufs, name='A', width_ufs=None)
B = te.placeholder((M, N), name='B')

def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: utils.ceilmult(m + 1, pad))
luf = len_ufw('s2k', 32).get_uf()

loop_ufs=[ls[0], ls[1]]
S = te.ragged_compute((M, N), [md, nd], loop_ufs,
                      lambda ds, rds: tvm.sum(tvm.tir.Cast('int32', rds['k'] < (ds[md] + 1)) *
                                              A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                              axis=rds['k'], dimensions = [kd]),
                      name = 'S', reduce_axis_ufs = [('k', luf)], width_uf_lists=None)

alpha = 2
O = te.ragged_compute((M, N), [md, nd], loop_ufs, lambda ds: alpha*S[ds[md], ds[nd]],
                      name = 'O', width_uf_lists=None)

s = tvm.create_schedule([O.op])

tl, to, tx, stl = 32, 64, 64, 4

################# Schedule ####################
S_l, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)
S_l_o_i, S_l_i = s[S].split(S_l, factor=stl)
S_l_o_o_i, S_l_o_i = s[S].split(S_l_o_i, factor=8)

S_k_o_o, S_k_o_i = s[S].split(S_k, factor=4)
s[S].reorder(S_l_o_o_i, S_k_o_o, S_k_o_i, S_l_o_i, S_o, S_l_i)
s[S].unroll(S_l_i)

O_l, O_o = tuple(O.op.axis)
O_l_o_i, O_l_i = s[O].split(O_l, factor=tl)

O_o_o_o_i, O_o_o_i = s[O].split(O_o, factor=to)
O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=2)

s[O].reorder(O_l_o_i, O_o_o_o_o, O_o_o_o_i, O_o_o_i, O_l_i)
s[S].compute_at(s[O], O_o_o_i)

A_shared = s.cache_read(A, "shared", [S])
A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
s[A_shared].compute_at(s[S], S_k_o_o)

B_shared = s.cache_read(B, "shared", [S], vanilla=True)
B_shared_ax0, B_shared_ax1 = tuple(B_shared.op.axis)
s[B_shared].compute_at(s[S], S_k_o_o)

s[O].bind(O_l_o_i, te.thread_axis("blockIdx.y"))
s[O].bind(O_o_o_o_o, te.thread_axis("blockIdx.x"))
s[O].bind(O_o_o_o_i, te.thread_axis("vthread"))
s[O].bind(O_o_o_i, te.thread_axis("threadIdx.x"))

A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=tx)
s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

B_shared_ax0_ax1_fused = s[B_shared].fuse(B_shared_ax0, B_shared_ax1)
B_shared_ax0_ax1_fused_o, B_shared_ax0_ax1_fused_i = s[B_shared].split(B_shared_ax0_ax1_fused, factor=4)
s[B_shared].vectorize(B_shared_ax0_ax1_fused_i)
B_shared_ax0_ax1_fused_o_o, B_shared_ax0_ax1_fused_o_i = s[B_shared].split(B_shared_ax0_ax1_fused_o, factor=tx)
s[B_shared].bind(B_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

################# Schedule ####################

inputs = [[], [A, B, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, run_function=run_utils.run_trmm,
                               prep_code_mode='no_prep_code')
