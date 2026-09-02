#include <stdio.h>

#define BLOCK 32

__global__ void matmul_kernel(float *a, float *b, float *c,
                              int m, int n, int k) {

  // map output data to threads
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  // shared mem
  __shared__ float a_s[BLOCK][BLOCK];
  __shared__ float b_s[BLOCK][BLOCK];

  // logic
  float acc = 0.0f;
  for (int ph = 0; ph < int(k / BLOCK); ph++) {

    // Collabarative loading of a and b
    a_s[ty][tx] = a[row * k + (ph * BLOCK + tx)];
    b_s[ty][tx] = b[(ph * BLOCK + ty) * m + col];
    }
    __syncthreads();
    for (int i = 0; i < BLOCK; i++) {
        acc += a_s[ty][i] * b_s[i][tx];
    }
  __syncthreads();

  if (row < m && col < n)
  c[row * n + col] = acc;
}

int main() {
int m = 1024;
  int n = 4096;
  int k = 2048;

  size_t size_a = m * k * sizeof(float);
  size_t size_b = k * n * sizeof(float);
  size_t size_c = m * n * sizeof(float);

  // Allocate host memory
  float *h_a = (float*)malloc(size_a);
  float *h_b = (float*)malloc(size_b);
  float *h_c = (float*)malloc(size_c);

  // Initialize host data
  for (int i = 0; i < m * k; i++) h_a[i] = 1.0f;
  for (int i = 0; i < k * n; i++) h_b[i] = 1.0f;

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  // Copy data host -> device
  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

  // Launch configuration
  dim3 grid((n + BLOCK - 1)/ BLOCK, (m + BLOCK -1)/ BLOCK);
  dim3 block(BLOCK, BLOCK);
  matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
// Copy result device -> host
  cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

  printf("Sample output C[0][0]: %f (Expected: %f)\n", h_c[0], (float)k);

  // Free memory
  free(h_a); free(h_b); free(h_c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;

}
