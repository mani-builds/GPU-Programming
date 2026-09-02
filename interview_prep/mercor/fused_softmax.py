#!/usr/bin/env python

import triton
import triton.language as tl
import torch

"""
Matrices whose rows can fit in the GPU’s SRAM
"""

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def naive_softmax(X: torch.Tensor):
    """
    X = (M, N)
    """
    M, N = X.shape
    # Read MN, write M
    row_max_x = X.max(dim=1)[0] #(M,)
    # Read MN and M, write MN
    z = X - row_max_x.reshape([M, 1]) #(M,N)
    # Read MN and write MN
    num = torch.exp(z)#(M,N)
    # Read MN and write M
    denominator = num.sum(dim=1) #(M,)
    # Read MN, M and write MN
    output = num / denominator.reshape([M, 1])

    # total reads = 5MN + 2M, total writes 3MN + 2M
    return output


@triton.jit
def fused_softmax_kernel(x_ptr, output_ptr,input_row_stride, output_row_stride,
                         n_row, n_col, BLOCK_SIZE:tl.constexpr,
                         num_stages:tl.constexpr = 4):
    # map data to programs
    row_start = tl.program_id(0) # equal to number of rows
    row_step = tl.num_programs(0)
    # Each program loads a set of rows of the input matrix X
    # strided by number of programs
    for row_start_idx in tl.range(row_start, n_row, row_step, num_stages=num_stages):

        row_ptr_start = x_ptr + row_start_idx * input_row_stride
        col_offsets =  tl.arange(0, BLOCK_SIZE) # block_size = col width
        mask = tl.arange(0, BLOCK_SIZE) < n_col

        # load data
        x = tl.load(row_ptr_start + col_offsets, mask=mask, other=-float('inf'))
        max = tl.max(x, axis=0)

        z = x - max

        num = tl.exp(z)
        denominator = tl.sum(num, axis=0)

        output = num / denominator

        output_ptrs = output_ptr + row_start_idx * output_row_stride

        # store data, write back to DRAM
        tl.store(output_ptrs + col_offsets, output, mask=mask)

def softmax_kernel(x: torch.Tensor):
    n_row, n_col = x.shape
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE

    BLOCK_SIZE = triton.next_power_of_2(n_col)
    # grid_size = triton.next_power_of_2(n_row)

    grid = (triton.cdiv(n_row, 2), )

    fused_softmax_kernel[grid](x, output, input_row_stride=x.stride(0),
                               output_row_stride=output.stride(0), n_row=n_row,
                               n_col=n_col, BLOCK_SIZE=BLOCK_SIZE)

    return output

torch.manual_seed(0)
M, N = 1024, 4096
x = torch.randn([M,N], device=DEVICE)
# output = torch.empty_like(x)
output_torch = naive_softmax(x)
output_triton = softmax_kernel(x)
print(output_torch)
print(output_triton)
# float32 default tolerances: rtol=1e-5, atol=1e-8
assert torch.allclose(output_triton, output_torch, atol=1e-8, rtol=1e-5)
print("SUCCESS: Triton output matches PyTorch!")
