
// matmul + bias + activation (relu)

#include <stdio.h>

#define TILE 32

__global__ void fused_matmul_bias_relu_kernel(float *a, float *b, float *c, float *bias, int M,
                             int N,
                             int K) {

  // Map output data to threads
  int row = blockIdx.y * TILE +  threadIdx.y;
  int col = blockIdx.x * TILE +  threadIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  // SRAM (move data to SRAM) and perform compuations
  __shared__ float as[TILE][TILE];
  __shared__ float bs[TILE][TILE];

  // Logic
  float pvalue = 0.0f;
  for (int ph = 0; ph < K / TILE; ph++) {
    as[ty][tx] = a[row * K + (ph * TILE + tx)];
    bs[ty][tx] = b[(ph * TILE + ty) * N + col];
    __syncthreads();

    for (int i = 0; i < TILE; i++) {
      pvalue += as[ty][i] * bs[i][tx];
    }
    __syncthreads();

  }
  // bias
  pvalue += bias[row];

  // activation
  if (pvalue < 0) pvalue = 0;
  // Write results back to HBM
  c[row * N + col] = pvalue;

}

// Launch config
dim3 blocks(TILE, TILE);
dim3 grid((N + blocks.x - 1) / blocks.x, (M + blocks.y - 1) / blocks.y);
