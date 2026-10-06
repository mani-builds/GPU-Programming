#include <cassert>
#include <stdio.h>
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

int main() {
  int n = 4096;
  float *a, *c;
  a = (float *)malloc(sizeof(float) * n);
  c = (float *)malloc(sizeof(float));
  for (int i = 0; i < n; i++) {
    a[i] = 1;
  }

  float sum=0;

  for (int i = 0; i < n; i++) {
    sum += a[i];
  }
  c[0] = 0;

  float *a_d, *c_d;
  cudaMalloc(&a_d, sizeof(float)*n);
  cudaMalloc(&c_d, sizeof(float));

  cudaMemcpy(a_d, a, sizeof(float)*n,cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c, sizeof(float),cudaMemcpyHostToDevice);

  dim3 threads(TILE, 1, 1);
  dim3 blocks((n + (2 * TILE) - 1) / (2 * TILE), 1, 1);
  sum_kernel<<<blocks, threads>>>(a_d, n, c_d);
  cudaMemcpy(c, c_d, sizeof(float),cudaMemcpyDeviceToHost);

  assert(sum == *c);
  // printf("Actual answer: %f\n", sum);
  // printf("Final answer: %f\n", *c);

  cudaFree(a_d);
  cudaFree(c_d);

  free(a);

}
