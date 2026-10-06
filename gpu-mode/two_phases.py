#!/usr/bin/env python

#!POPCORN leaderboard vectorsum_v2
#!POPCORN gpu A100

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# High-performance vector sum reduction using CUDA.
# Strategy:
#   1. Vectorized float4 loads to maximize memory bandwidth
#   2. Thread coarsening: each thread processes multiple float4 chunks
#   3. Warp-level shuffle reduction (no shared memory bank conflicts)
#   4. Block-level reduction via shared memory
#   5. Two-phase: phase 1 produces partial sums, phase 2 reduces them
#
# Tuned for A100 (108 SMs, 2 TB/s HBM2e bandwidth, 40MB L2).

cuda_source = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Phase 1: Each block reduces a large chunk of the input into one partial sum.
// Uses float4 vectorized loads and thread coarsening.
__global__ void reduce_phase1(
    const float* __restrict__ input,
    float* __restrict__ partial_sums,
    int n
) {
    // Shared memory for block-level reduction (one slot per warp)
    __shared__ float smem[32];  // max 32 warps per block

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x * block_size;

    float sum = 0.0f;

    // Vectorized float4 loads: each thread processes 4 floats at a time
    // with grid-stride loop for coarsening
    const int n4 = n / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input);

    for (int i = bid * block_size + tid; i < n4; i += grid_size) {
        float4 v = input4[i];
        sum += v.x + v.y + v.z + v.w;
    }

    // Handle remaining elements (n % 4 tail)
    int tail_start = n4 * 4;
    for (int i = tail_start + bid * block_size + tid; i < n; i += grid_size) {
        sum += input[i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Write warp results to shared memory
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (lane == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces all warp partial sums
    const int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }

    // Thread 0 writes the block's partial sum
    if (tid == 0) {
        partial_sums[bid] = sum;
    }
}

// Phase 2: Reduce partial sums into a single scalar.
// Single block, single warp is enough for small number of partial sums.
__global__ void reduce_phase2(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    int n
) {
    __shared__ float smem[32];

    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        sum += partial_sums[i];
    }

    sum = warp_reduce_sum(sum);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (lane == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }

    if (tid == 0) {
        output[0] = sum;
    }
}

torch::Tensor vector_sum_cuda(torch::Tensor input, torch::Tensor output) {
    const int n = input.numel();

    // A100: 108 SMs. We want enough blocks to saturate all SMs
    // but not so many that phase 2 becomes expensive.
    // 256 threads/block, ~432 blocks gives good occupancy on A100.
    const int threads = 256;
    const int blocks = min(432, (n / 4 + threads - 1) / threads);

    // Allocate temporary buffer for partial sums
    auto partial_sums = torch::empty({blocks}, input.options());

    reduce_phase1<<<blocks, threads>>>(
        input.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n
    );

    // Phase 2: reduce partial sums (blocks <= 432, one block of 256 threads is plenty)
    reduce_phase2<<<1, 256>>>(
        partial_sums.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );

    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor vector_sum_cuda(torch::Tensor input, torch::Tensor output);
"""

# Compile with optimizations
module = load_inline(
    name="vector_sum_fast",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["vector_sum_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    module.vector_sum_cuda(data, output)
    return output[0]
