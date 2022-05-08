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
assert args.target == 'cuda'
# target = "llvm"

BATCH_SIZE = te.var('bs')
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
print("get MAX_LEN:", MAX_LEN)
lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')
E_SIZE = 1024
bd, ld = Dim('bd'), Dim('ld')
ed = Dim('ed')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens],
                                   lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw32 = len_ufw('s64', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: lufw1.get_uf(),
    2: lufw32.get_uf(),
    3: Uf.from_constant('ed', E_SIZE, 'l'),
}

loop_ufs=[ls[0], ls[1], ls[3]]
width_ufs=[ls[0], ls[2], ls[3]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, E_SIZE), [bd, ld, ed], loop_ufs, name='A', width_ufs=width_ufs)

O = te.ragged_compute((BATCH_SIZE, MAX_LEN, E_SIZE), [bd, ld, ed], loop_ufs,
            lambda ds: 2 * A[ds[bd], ds[ld], ds[ed]], name = 'O', width_uf_lists=[width_ufs])

s = tvm.create_schedule([O.op])

thread_x = tvm.thread_axis("threadIdx.x")
block_x = tvm.thread_axis("blockIdx.x")

xo, xi, xj = s[O].leaf_iter_vars
f = s[O].fuse(xo, xi, padding=64)
fo, fi = s[O].split(f, factor = 2)
s[O].bind(fo, block_x)
s[O].bind(fi, thread_x)

inputs = [[lens], [BATCH_SIZE, A, O]]


def size_fn(l_inputs):
    lens = l_inputs[0] # batch [1, 2, 4, 5]
    fn = lufw32.get_fn(lens)
    return {
        A: run_utils.prefix_sum(len(lens), lambda b: (fn(b)*E_SIZE)),
        O: run_utils.prefix_sum(len(lens), lambda b: (fn(b)*E_SIZE))
    }

prep_code_mode = 'with_prep_code'
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                                        run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE),
                                        prep_code_mode=prep_code_mode, pad_sum=64)
