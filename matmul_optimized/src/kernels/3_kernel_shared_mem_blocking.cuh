#include <cstdio>

// sgemm performs C = alpha * AB + beta * C

/*

Matrix sizes:
MxK * KxN = MxN

*/


#define BLOCKSIZE 32

__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, float *a, float *b,  
                            float beta, float *c) {

  // Map output data to thread idx
  int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
  int col = blockIdx.x * BLOCKSIZE + threadIdx.x;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  // smem
  __shared__ float a_s[BLOCKSIZE][BLOCKSIZE];
  __shared__ float b_s[BLOCKSIZE][BLOCKSIZE];

  // Logic implementation
  if (row < M && col < N){ //boundary conditions
    float acc = 0;
    for (int ph = 0; ph < K / BLOCKSIZE; ph++) {
      a_s[ty][tx] = a[row * K + (ph * BLOCKSIZE * tx)];
      b_s[ty][tx] = b[(ph * BLOCKSIZE * ty) * K + col];
      __syncthreads();
      for (int i = 0; i < BLOCKSIZE; i++){
        acc += a_s[ty][i] + b_s[i][tx];
      }
      __syncthreads();
    }
    // Write to HBM
    c[row * N + col] = alpha * acc + beta * c[row * N + col];
  }
}
