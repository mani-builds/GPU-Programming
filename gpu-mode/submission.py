#!/usr/bin/env python

from task import input_t, output_t

import torch
from torch.utils.cpp_extension import load_inline
device = "cuda" if torch.cuda.is_available() else "cpu"

_CPP_SOURCE=r"""
#include <torch/extension.h>

torch::Tensor vectorSum(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("vectorSum", &vectorSum, "Sum reduction using CUDA kernel");
}

"""

_CUDA_SOURCE=r"""
#include <cassert>
#include <stdio.h>
#include <typeinfo>
#define TILE 1024

__global__ void sum_kernel(const float* __restrict__ a, const int n, float* __restrict__ c) {
    // Map data to threads
    int i = 2*blockIdx.x * TILE +  threadIdx.x;
   int tid = threadIdx.x;

    // SMEM
    __shared__ float a_s[TILE];

    float sum = 0.0f;
    if (i + TILE < n) {
      sum = a[i] + a[i + TILE];
    } else if (i < n) {
      sum = a[i];
    }
    a_s[tid] = sum;
    __syncthreads();

    // logic implement
    for (int stride = TILE / 2; stride >= 1; stride /= 2) {
      if (threadIdx.x < stride) {
        a_s[threadIdx.x] = a_s[threadIdx.x] + a_s[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // write to HBM
    if (threadIdx.x == 0){
      atomicAdd(c, a_s[0]);
    }
}

torch::Tensor vectorSum(torch::Tensor input, torch::Tensor out){
    auto options = torch::TensorOptions().device(torch::kCUDA, 0);
    TORCH_CHECK(input.is_cuda(), "input must be cuda tensors");
    input = input.contiguous();
    const int n = input.numel();

    out.zero_();
    TORCH_CHECK(out.is_cuda(), "out must be cuda tensors");
    // printf("Everything is fine.");

    dim3 threads(TILE, 1, 1);
    dim3 blocks((n + (2 * TILE) - 1) / (2 * TILE), 1, 1);
    sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        n, out.data_ptr<float>());


    // synchronize to catch device errors before reading
    cudaDeviceSynchronize();

    // torch::Tensor expected = input.sum();
    // TORCH_CHECK(torch::allclose(out, expected, 1e-4, 1e-6),
    //        "vectorSum mismatch: exptected, ", expected.item(),
    //          " got ", out.item<float>());

    return out;
    }

"""

_EXT = load_inline(
    name="vectorSum",
    cpp_sources=[_CPP_SOURCE],
    cuda_sources=[_CUDA_SOURCE],
    functions=None,
    # extra_cflags=["-O3 -use_fast_math"],
    # extra_cuda_cflags=["-O3 -use_fast_math"],
    with_cuda=True,
    verbose=False,
)

def custom_kernel(data: input_t) -> output_t:
    data, output = data
    out = _EXT.vectorSum(data, output)
    # output = torch.zeros((1,), dtype = torch.float32, device="cuda")
    # output = _EXT.vectorSum(data, output)
    return out

def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensor of specified shape with random offset and scale.
    The data is first generated as standard normal, then scaled and offset
    to prevent trivial solutions.

    Returns:
        Tensor to be reduced
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    # Generate base random data
    data = torch.randn(
        size, device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()

    # Generate random offset and scale (using different seeds to avoid correlation)
    offset_gen = torch.Generator(device="cuda")
    offset_gen.manual_seed(seed + 1)
    scale_gen = torch.Generator(device="cuda")
    scale_gen.manual_seed(seed + 2)

    # Generate random offset between -100 and 100
    # offset = (torch.rand(1, device="cuda", generator=offset_gen) * 200 - 100).item()
    # Generate random scale between 0.1 and 10
    # scale = (torch.rand(1, device="cuda", generator=scale_gen) * 9.9 + 0.1).item()

    # Apply scale and offset
    # input_tensor = (data * scale + offset).contiguous()
    input_tensor = data
    output_tensor = torch.empty(1, device="cuda", dtype=torch.float32)
    return input_tensor, output_tensor


n_elem = 52428800
seed = 123
torch.manual_seed(seed)
input = torch.randn(n_elem, device=device, dtype=torch.float32)
custom_kernel(generate_input(n_elem, seed))
