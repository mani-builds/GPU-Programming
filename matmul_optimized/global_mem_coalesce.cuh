#include <cstdio>

// sgemm performs C = alpha * AB + beta * C

/*

Matrix sizes:
MxK * KxN = MxN

*/


#define BLOCK_SIZE 1024

__global__ void sgemm_global_mem_coalesce(float *a, float *b, float *c, int M, int K, int N,
                            int alpha, int beta) {

  // Map output data to thread idx
  // Select one row from A and all the cols 1-by-1 to exploit coalescing in same warp 
  int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
  int col = blockIdx.x * BLOCKSIZE + threadIdx.x;
 
  // smem
  // Logic implementation
  if (row < M && col < N){ //boundary conditions
    float acc = 0;
    for (int i = 0; i < K; i++) {
        acc += a[row * K + i] + b[i * N + col];
    }
    // Write to HBM
    c[row * N + col] = alpha * acc + beta * c[row * N + col];
  }
}

