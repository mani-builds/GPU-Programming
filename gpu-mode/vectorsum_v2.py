#!/usr/bin/env python

import torch
from torch.utils.cpp_extension import load_inline
device = "cuda" if torch.cuda.is_available() else "cpu"

_CPP_SOURCE=r"""
#include <torch/extension.h>

void vectorSum(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("vectorSum", &vectorSum, "Sum reduction using CUDA kernel");
}

"""

_CUDA_SOURCE=r"""
#include <cassert>
#include <stdio.h>
#include <typeinfo>
#define TILE 1024

__global__ void sum_kernel(float *a, int n, float *c) {
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
    if (threadIdx.x == 0)
      atomicAdd(c, a_s[0]);
}

void vectorSum(torch::Tensor input){
    auto options = torch::TensorOptions().device(torch::kCUDA, 0);
    TORCH_CHECK(input.is_cuda(), "input must be cuda tensors");
    input = input.contiguous();
    int n = input.size(0);

    // float sum = input.sum();
    torch::Tensor out;
    out = torch::zeros(1, options);
    TORCH_CHECK(out.is_cuda(), "out must be cuda tensors");
    // printf("Everything is fine.");

    dim3 threads(TILE, 1, 1);
    dim3 blocks((n + (2 * TILE) - 1) / (2 * TILE), 1, 1);
    sum_kernel<<<blocks, threads>>>(
        reinterpret_cast<float *>(input.data_ptr()),
        n, reinterpret_cast<float *>(out.data_ptr()));


    // synchronize to catch device errors before reading
    cudaDeviceSynchronize();

    torch::Tensor expected = input.sum();
    TORCH_CHECK(torch::allclose(out, expected, 1e-4, 1e-6),
            "vectorSum mismatch: exptected, ", expected.item<float>(),
            " got ", out.item<float>());
    }
"""

_EXT = load_inline(
    name="vectorSum",
    cpp_sources=[_CPP_SOURCE],
    cuda_sources=[_CUDA_SOURCE],
    functions=None,
    extra_cflags=["-O3 -use_fast_math"],
    extra_cuda_cflags=["-O3 -use_fast_math"],
    with_cuda=True,
    verbose=False,
)
def custom_kernel(input: torch.Tensor):

    kernel = _EXT.vectorSum(input)
    pass


n_elem = 1638400
input = torch.ones(n_elem, device=device)
custom_kernel(input)
