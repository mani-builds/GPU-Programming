#include <__clang_cuda_builtin_vars.h>
#include <iostream>

#define BLOCK 32

__global__ void matmul_kernel(float *a, float *b, float *c, int M, int K, int N) {

  // Map threads to data
  int row = blockIdx.y * BLOCK +  threadIdx.y;
  int col = blockIdx.x * BLOCK +  threadIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // shared mem
  __shared__ float as[BLOCK][BLOCK];
  __shared__ float bs[BLOCK][BLOCK];

  // Logic
  float pvalue = 0.0;
  for (int ph = 0; ph < K / BLOCK; ph++) {
    // Collaboratively load data into block
    as[ty][tx] = a[row * M + (ph * BLOCK) + tx];
    bs[ty][tx] = b[(ph * BLOCK + ty) * N  + col];
    __syncthreads();

    for (int i = 0; i < BLOCK; i++) {
     pvalue += as[ty][i] * bs[i][tx];
    }
    __syncthreads();
  }
  //HBM
  b[row * N + col] = pvalue;
  }
dim3 threads(32, 32);
dim3 blocks(N, M);
