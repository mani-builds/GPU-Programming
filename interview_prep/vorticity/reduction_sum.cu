#include <__clang_cuda_builtin_vars.h>
#include <iostream>
#include <cstdio>

using namespace std;

#define BLOCK 512

__global__ void reduction(float *arr, float *sum, int N) {

  // Map data to threads
  int i = 2 * blockIdx.x * BLOCK + threadIdx.x;

  // smem
  __shared__ float arr_s[BLOCK];
  float temp = 0.0f;
    if (i < N) temp += arr[i];
    if (i + BLOCK < N) temp += arr[i + BLOCK];
    arr_s[threadIdx.x] = temp;
   __syncthreads();

  // reduction
  float acc = 0.0;
  for (int k = BLOCK/2; k >= 1; k /= 2) {
    // enable first-half of threads only
    if (threadIdx.x < k) {
      arr_s[threadIdx.x] = arr_s[threadIdx.x] + arr_s[threadIdx.x + k];
    }
  __syncthreads();
  }
  // HBM
  if (threadIdx.x == 0) {
    atomicAdd(sum, arr_s[threadIdx.x]);
  }
}

// Launch
int threads = BLOCK;
int blocks = ceil_div(N / BLOCK / 2);
