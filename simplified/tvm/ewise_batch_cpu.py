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

BATCH_SIZE = te.var('bs')
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
print("get MAX_LEN:", MAX_LEN)
lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd, ld = Dim('bd'), Dim('ld')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens],
                                   lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw32 = len_ufw('s64', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: lufw1.get_uf(),
    2: lufw32.get_uf(),
}

loop_ufs=[ls[0], ls[1]]
width_ufs=[ls[0], ls[2]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN), [bd, ld], loop_ufs, name='A', width_ufs=width_ufs)

O = te.ragged_compute((BATCH_SIZE, MAX_LEN), [bd, ld], loop_ufs,
            lambda ds: 2 * A[ds[bd], ds[ld]], name = 'O', width_uf_lists=[width_ufs])

s = tvm.create_schedule([O.op])

xo, xi = s[O].leaf_iter_vars
f = s[O].fuse(xo, xi, padding = 64)
fo, fi = s[O].split(f, factor = 64)
inputs = [[lens], [BATCH_SIZE, A, O]]


def size_fn(l_inputs):
    lens = l_inputs[0] # batch [1, 2, 4, 5]
    fn = lufw32.get_fn(lens)
    return {
        A: run_utils.prefix_sum(len(lens), lambda b: (fn(b))),
        O: run_utils.prefix_sum(len(lens), lambda b: (fn(b)))
    }

prep_code_mode = 'with_prep_code'
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                                        run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE),
                                        prep_code_mode=prep_code_mode, pad_sum=64)
