#include <__clang_cuda_builtin_vars.h>
#include <iostream>
#include <cstdio>

using namespace std;

/* 9-point 2D stencil */

#define RADIUS 2
#define IN_TILE_WIDTH 8
#define OUT_TILE_WIDTH ((IN_TILE_WIDTH) - (2 * RADIUS))

__constant__ float FILTER[9];

__global__ void stencil(float *a, float *b, int M, int N) {
   // Map threads to data
   int tx = threadIdx.x;
   int ty= threadIdx.y;
// By subtracting the RADIUS, you shift the global memory lookup "up and to the left."
// This forces the threads on the outer edges of your block
// to automatically grab the halo elements.
  int col = blockIdx.x * OUT_TILE_WIDTH + tx - RADIUS;
  int row = blockIdx.y * OUT_TILE_WIDTH + ty - RADIUS;

  // smem
  __shared__ float a_s[IN_TILE_WIDTH][IN_TILE_WIDTH];
  if (col >= 0 && col < N && row >= 0 && row < M){
    // gaurd againt halo
    // ty and tx help grad all elements in IN_TILE_WIDTH
    a_s[ty][tx] = a[row * N + col];
  } else {
    a_s[ty][tx] = 0.0f;
  }
  __syncthreads();
  // matmul
  // gaurd againt edge rows and cols
  if (col >= RADIUS && col < N - RADIUS && row >= RADIUS && row < N - RADIUS) {
    // turn off threads that are outside of OUT_TILE_WIDTH
    if (tx >= RADIUS && tx < OUT_TILE_WIDTH && ty >= RADIUS && ty < OUT_TILE_WIDTH) {
      b[row * N + col] = FILTER[0] * a_s[ty][tx] +
                         FILTER[1] * a_s[ty][tx-1] +FILTER[2] * a_s[ty][tx + 1] +
                         FILTER[3] * a_s[ty][tx-2] +FILTER[4] * a_s[ty][tx + 2] +
                         FILTER[6] * a_s[ty - 1][tx] + FILTER[6] * a_s[ty + 1][tx] +
                         FILTER[7] * a_s[ty - 2][tx] + FILTER[8] * a_s[ty + 2][tx];

    }
  }
}


dim3 blocks(IN_TILE_WIDTH, IN_TILE_WIDTH);
dim3 grid(ceil_div(N/OUT_TILE_WIDTH), ceil_div(M/OUT_TILE_WIDTH));
