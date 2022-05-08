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
            name = 'O', reduce_axis_ufs = [('k', ls[2])], width_uf_lists=None)


block_size = 16  # the number of threads for one dimension in a thread block.
tx, ty, tk = 8, 4, 32  # tile sizes for one CUDA thread

s = tvm.create_schedule([C.op])

inputs = [[lens], [A, B, C]]


# Save into the d2ltvm package.
def split(stage, axis, factors):
    """Split an axis by a list of factors in a reverse order
    """
    axes = []
    for f in reversed(factors):
        axis, x = stage.split(axis, f)
        axes.append(x)
    return list(reversed(axes+[axis]))

# Save into the d2ltvm package.
def bind_thread(stage, axes, tags):
    """Bind a list of axes to thread axes
    """
    for axis, tag in zip(axes, tags):
        stage.bind(axis, te.thread_axis(tag))

# Create caches
A_shared = s.cache_read(A, "shared", [C])
A_local  = s.cache_read(A_shared, "local", [C])
B_shared = s.cache_read(B, "shared", [C])
B_local  = s.cache_read(B_shared, "local", [C])
C_local = s.cache_write(C, "local")
# Split each axis into block axis, thread axis, and inner axis
b, x, y = s[C].op.axis
xb, xo, xi = split(s[C], x, (block_size, tx))
yb, yo, yi = split(s[C], y, (block_size, ty))
s[C].reorder(b, xb, yb, xo, yo, xi, yi)

bxb = s[C].fuse(b, xb)


# Note that we bind yb to blockIdx.x instead of blockIdx.y
bind_thread(s[C], (yb, bxb, yo, xo),
            ("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y"))
# Schedule C_local
s[C_local].compute_at(s[C], yo)
b, yi, xi = s[C_local].op.axis
k, = s[C_local].op.reduce_axis
ko, ki = s[C_local].split(k, tk)
s[C_local].reorder(ko, ki, yi, xi)
# Optimize read caches of A and B with cooperative fetching
def optimize_read_cache(shared, local):
    s[shared].compute_at(s[C_local], ko)
    s[local].compute_at(s[C_local], ki)
    b, y, x = s[shared].op.axis
    # Note that we must split into block_size parts to reuse
    # the previous axis threads
    yo, yi = s[shared].split(y, nparts=block_size)
    xo, xi = s[shared].split(x, nparts=block_size)
    s[shared].reorder(yo, xo, yi, xi)
    bind_thread(s[shared], (yo, xo), ("threadIdx.y", "threadIdx.x"))
optimize_read_cache(A_shared, A_local)
optimize_read_cache(B_shared, B_local)

def size_fn(l_inputs):
    lens = l_inputs[0] # batch [1, 2, 4, 5]
    fn = lufw32.get_fn(lens)
    return {
        A: run_utils.prefix_sum(len(lens), lambda b: (fn(b)*E_SIZE)),
        C: run_utils.prefix_sum(len(lens), lambda b: (fn(b)*V_SIZE))
    }

prep_code_mode = 'with_prep_code'

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                                        run_function=run_utils.get_gemm_run_fn(BATCH_SIZE),
                                        prep_code_mode=prep_code_mode, pad_sum=64)
