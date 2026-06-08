#include <__clang_cuda_builtin_vars.h>
#include <iostream>

#define BLOCK 32

__global__ void transpose_kernel(float *a, float *b, int M, int N) {

  // Map threads to data
  int row_in = blockIdx.y * BLOCK +  threadIdx.y;
  int col_in = blockIdx.x * BLOCK +  threadIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // shared mem
  __shared__ float as[BLOCK][BLOCK];
  __shared__ float bs[BLOCK][BLOCK];

  // Logic
  as[ty][tx] = a[row_in * N + col_in];
  __syncthreads();

  // HBM
  int row_out = bx * BLOCK + ty;
  int col_out = by * BLOCK + tx;
  b[row_out * M + col_out] = as[tx][ty];
}
dim3 threads(32, 32);
dim3 blocks(N, M);
