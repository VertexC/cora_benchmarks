import argparse
import numpy as np
import time
import torch
import torch.nn.functional as f
from torch.profiler import profile, record_function, ProfilerActivity
from torch import Tensor
from torch import nn
import torch.utils.benchmark as benchmark
import sys
sys.path.append("../")
import run_utils
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--profile', dest='profile', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random_384_512')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

def get_np_tensor(size, device, random, fill_value = None):
    if random: return torch.randn(size, device = device, requires_grad = False, dtype = torch.float32)
    else:
        if fill_value == None: raise ValueError("No fill value provided " + str(fill_value))
        return torch.full(size, fill_value = fill_value, device = device, requires_grad = False, dtype = torch.float32)

class Encoder(nn.Module):
    def __init__(self, device, max_len, batch_size, num_heads, head_size, model_size, ff_size):
        super(Encoder, self).__init__()
        self.pre_linear_w = get_np_tensor((3, num_heads, model_size, head_size), device, True)
        self.pre_linear_b = get_np_tensor((3, num_heads, model_size), device, True)
        self.post_linear_w = get_np_tensor((model_size, model_size), device, True)
        self.post_linear_b = get_np_tensor((model_size,), device, True)
        self.ff1_w = get_np_tensor((model_size, ff_size), device, True)
        self.ff2_w = get_np_tensor((ff_size, model_size), device, True)
        self.ff1_b = get_np_tensor((ff_size,), device, True)
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.model_size = model_size
        self.ff_size = ff_size
        self.max_len = max_len

    def forward(self, inp):
        qkv = torch.matmul(inp, self.pre_linear_w) + self.pre_linear_b
        qkv = qkv.view(3, self.num_heads, self.batch_size, self.max_len, self.head_size)
        q, k, v = torch.split(qkv, 1, 0)
        attn = torch.matmul(q, k.permute(0, 1, 2, 4, 3))
        attn = f.softmax(attn, dim = 4)
        attn = torch.reshape(torch.matmul(attn, v).permute(0, 2, 3, 1, 4), (self.batch_size, self.max_len, self.model_size))
        sa_out = torch.matmul(attn, self.post_linear_w) + self.post_linear_b
        sa_out = f.layer_norm(sa_out + inp.view(self.batch_size, self.max_len, self.model_size),
                              normalized_shape = (self.model_size,))

        ff1_out = f.relu(torch.matmul(sa_out, self.ff1_w) + self.ff1_b)
        ff2_out = torch.matmul(ff1_out, self.ff2_w)
        ff_out = f.layer_norm(ff2_out + sa_out, normalized_shape = (self.model_size,))
        return ff_out

num_heads = 8
head_size = 64
ff_size = 2048
max_len = 512
batch_size = 32
model_size = num_heads * head_size
device = torch.device('cuda')

batches = run_utils.get_nlp_batches(args.batch_size, args.max_batches, args.dataset, args.datadir)

callable_to_profile = None
def run_for_batches():
    batch_times = []
    for batch in batches:
        max_len = int(np.amax(batch))

        encoder = Encoder(device, max_len, batch_size, num_heads, head_size, model_size, ff_size)
        traced_encoder = torch.jit.script(encoder)

        inp = get_np_tensor((batch_size * max_len, model_size), device, True)
        timer = benchmark.Timer(stmt='f(x)',
                                globals={'x': inp, 'f': traced_encoder})
        batch_times.append(timer.timeit(50).mean * 1000.0)
    return batch_times

if not args.profile:
    batch_times = run_for_batches()
    print(sum(batch_times) / len(batches))
else:
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        run_for_batches()
        print(prof.key_averages(group_by_stack_n=5))
