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

parser = run_utils.get_cmd_parser()
parser.add_argument('--nt', dest='nt', default=8, type=int)
parser.add_argument('--kt', dest='kt', default=4, type=int)
parser.add_argument('--masked-mha', dest='masked_mha', default=False, action='store_true')
parser.add_argument('--sched', dest='sched', default=1, type=int)
args = parser.parse_args()

args.target = run_utils.get_arm_target()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qk = Dim('qk')
bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
# if args.sched == 1: lufw = len_ufw('s', 64)
# if dataset in ['cola', 'mrpc']: lufw = len_ufw('s', 64)
# else: lufw = len_ufw('s', 32)
lufw = len_ufw('s', 64)
sufw = len_ufw('s', 64)

lbduf = Uf.from_constant('bd', BS_VAR, "l")
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, "l"),
    1: Uf.from_constant('md', NUM_HEADS, "l"),
    2: lufw.get_uf(),
    3: lufw.get_uf(),
    4: Uf.from_constant('hd', HEAD_SIZE, "l"),
    5: Uf.from_constant('qk', 3, "l"),
}

loop_ufs=[ls[5], ls[0], ls[2], ls[1], ls[4]]
width_ufs = None if args.dense_storage else [ls[5], ls[0], sufw.get_uf(), ls[1], ls[4]]
Q = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[ls[5], ls[0], ls[3], ls[1], ls[4]]
width_ufs = None if args.dense_storage else [ls[5], ls[0], sufw.get_uf(), ls[1], ls[4]]
K = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s2, md, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[lbduf, ls[2], ls[1], ls[3]]
width_ufs = None if args.dense_storage else [[ls[0], sufw.get_uf(), ls[1], sufw.get_uf()]]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[0, ds[bd], ds[s1], ds[md], k] * K[1, ds[bd], ds[s2], ds[md], k],
                                         axis = k, dimensions=[hd]),
                      name = 'S', width_uf_lists=width_ufs)

def get_threshold(ds):
    if args.masked_mha:
        return ds[s1] + 1
    else:
        return lens[ds[bd]]

O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.if_then_else(ds[s2] >= get_threshold(ds), -float('inf'), S[ds[bd], ds[s1], ds[md], ds[s2]]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.sched == 1:
    O_local = S

    Ks = s.cache_read(K, "shared", [S], layouts='dense')

    O_local_b_c, O_local_m_c, O_local_h_c, O_local_n_c, O_local_k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)

    O_local_m_c_o_i, O_local_m_c_i = s[O_local].split(O_local_m_c, factor=2)
    O_local_m_c_o_o_i, O_local_m_c_o_i = s[O_local].split(O_local_m_c_o_i, factor=1)
    O_local_m_c_o_o_o, O_local_m_c_o_o_i = s[O_local].split(O_local_m_c_o_o_i, factor=8)

    O_local_n_c_o_i, O_local_n_c_i = s[O_local].split(O_local_n_c, factor=8)

    O_local_k_o, O_local_k_i = s[O_local].split(O_local_k, factor=64)
    O_local_k_i_o, O_local_k_i_i = s[O_local].split(O_local_k_i, factor=16)
    s[O_local].reorder(O_local_b_c, O_local_m_c_o_o_o, O_local_n_c_o_i, O_local_m_c_o_o_i, O_local_k_o, O_local_m_c_o_i, O_local_k_i_o, O_local_k_i_i, O_local_m_c_i, O_local_n_c_i)

    s[O_local].unroll(O_local_k_i_i)
    s[O_local].unroll(O_local_n_c_i)
    s[O_local].unroll(O_local_m_c_i)

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)
    s[O].reorder(b, xo, yo, h, xi, yi)
    f1 = s[O].fuse(b, xo)
    f1 = s[O].fuse(f1, yo)
    f1 = s[O].fuse(f1, h)
    s[O].parallel(f1)

    O_m, O_n = xi, yi

    O_m_o_i, O_m_i = s[O].split(O_m, factor=2)
    O_m_o_o, O_m_o_i = s[O].split(O_m_o_i, factor=8)

    O_n_o_i, O_n_i = s[O].split(O_n, factor=8)
    s[O].reorder(O_m_o_o, O_n_o_i, O_m_o_i, O_m_i, O_n_i)
    s[S].compute_at(s[O], O_n_o_i)
    s[Ks].compute_at(s[O], O_n_o_i)

    s[O_local].vectorize(O_local_n_c_i)
    s[O].vectorize(O_n_i)

    s.reorder_tensor_dimensions(Ks, 2, 3)
    s.reorder_tensor_dimensions(Ks, 3, 4)
else:
    Ks = s.cache_read(K, "local", [S], layouts='dense')
    Qs = s.cache_read(Q, "local", [S], layouts='dense')
    S_b_c, S_l1_c, S_n_c, S_l2_c, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)

    S_l1_c_o_o_i, S_l1_c_o_i = s[S].split(S_l1_c, factor=8)
    S_l2_c_o_i, S_l2_c_i = s[S].split(S_l2_c, factor=8)
    s[S].reorder(S_b_c, S_n_c, S_l2_c_o_i, S_l1_c_o_o_i, S_k, S_l1_c_o_i, S_l2_c_i)
    s[Ks].compute_at(s[S], S_k)
    s[Qs].compute_at(s[S], S_k)

    O_b, O_l1, O_n, O_l2 = tuple(O.op.axis) + tuple(O.op.reduce_axis)

    O_l1_o_i, O_l1_i = s[O].split(O_l1, factor=8)
    O_l1_o_o, O_l1_o_i = s[O].split(O_l1_o_i, factor=8)

    O_l2_o_i, O_l2_i = s[O].split(O_l2, factor=8)
    O_l2_o_o, O_l2_o_i = s[O].split(O_l2_o_i, factor=1)

    s[O].reorder(O_b, O_l1_o_o, O_n, O_l2_o_o, O_l1_o_i, O_l2_o_i, O_l1_i, O_l2_i)
    s[S].compute_at(s[O], O_l2_o_i)
    O_b_o_o_l1_o_o_fused_n_o_o_fused = s[O].fuse(O_b, O_l1_o_o, O_n)
    s[O].parallel(O_b_o_o_l1_o_o_fused_n_o_o_fused)

    s[S].pragma(S_b_c, "auto_unroll_max_step", 512)
    s[S].pragma(S_b_c, "unroll_explicit", True)
    s[S].vectorize(S_l2_c_i)
    s[O].vectorize(O_l2_i)

inputs = [[lens], [BS_VAR, Q, K, O]]

def size_fn(l_inputs):
    lens = l_inputs[0]
    if args.dense_storage: return {}
    return {
        Q: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (sufw.get_fn(lens)(b))),
        K: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (sufw.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (sufw.get_fn(lens)(b) *
                                                       sufw.get_fn(lens)(b)))
    }

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64, hoist_loads=True,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))


# _, Q, K, O = out[0:4]
# O = O.flatten()
# ctr = 0
# for length in batches[0]:
#     rounded = utils.ceilmult(length, TILE)
#     this_extent = rounded
#     this_storage_extent = rounded * rounded * NUM_HEADS
#     # print(rounded, np.mean(O[ctr:ctr+this_storage_extent]))
#     print(rounded, np.mean(O[ctr:ctr+length]))
#     ctr += this_storage_extent
