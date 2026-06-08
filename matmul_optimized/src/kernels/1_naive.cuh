
#include <cstdio>

// sgemm performs C = alpha * AB + beta * C

/*

Matrix sizes:
MxK * KxN = MxN

*/


#define BLOCKSIZE 32

__global__ void sgemm_naive(int M, int N, int K, float alpha, float *a, float *b,  
                            float beta, float *c) {

  // Map output data to thread idx
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

