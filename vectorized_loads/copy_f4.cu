#include <cassert>
#include <numeric>
#include <stdio.h>
#include <iostream>
#include <vector>

__global__ void copy(float* in, float* out, int N){
  // "grid-strip loop", it helps if there are more arrays than the total
  // threads the CUDA device can handle.
  // blockDim.x * gridDim.x = all the threads in a grid
  // If total threads in a grid is 1028
  // thread = 0 handles array indices 0, 1280, 2560,etc.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N; i += blockDim.x * gridDim.x){
    out[i] = in[i];
  }
}

__global__ void copy_vectorized(float* in, float* out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // now the same copy kernel as before, but now with vectozied float4
  // One keep requirement for vectorized floats is the pointer (address
  // of the variable) has to be aligned with the size of float4 (128-byte).
  // Device-allocated memory is automatically aligned to a multiple of the size of the data type.
  for (int i = idx; i < N/4;
       i += blockDim.x * gridDim.x) {
    reinterpret_cast<float4*>(out)[i] = reinterpret_cast<float4*>(in)[i];
  }

  // (if N is not a multiple of 4), then handle remaning elements(0-3)
  // in parallel with first few threads
  int remainder = N % 4;
  int remainder_start = N - remainder;
  if (idx < remainder){
    out[remainder_start + idx] = in[remainder_start + idx];
  }
}

int main() {

  int n = 4096*4096 + 4;
  float *in_h;
  float *ou_h;
  in_h = (float *) malloc(sizeof(float) * n);
  ou_h = (float *) malloc(sizeof(float) * n);

  for(int i=0; i < n; i++) in_h[i] = 1.0f;

  float *in, *out;
  cudaMalloc(&in, sizeof(float)* n);
  cudaMalloc(&out, sizeof(float)* n);
  cudaMemcpy(in, in_h, sizeof(float)*n, cudaMemcpyHostToDevice);

  int devId;
  cudaGetDevice(&devId);

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, devId);
  std::cout<< "maxThreadsPerBlock : " << prop.maxThreadsPerBlock <<
  "\nmaxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor
  <<"\n maxBlocksPerMultiProcessormax: " << prop.maxBlocksPerMultiProcessor
  <<"\n maxThreadsPerBlock: " << prop.maxThreadsPerBlock <<
  "\nmultiProcessorCount: " << prop.multiProcessorCount <<
  "\nTotal physical threads on the device: " << prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount
  << std::endl;

  int threads_per_block = 1024;
  int blocks_per_grid = 32*numSMs; // it's efficient if blocks are a multiple of SMs
  // copy<<<blocks_per_grid, threads_per_block>>>(in, out, n);
  copy_vectorized<<<blocks_per_grid, threads_per_block>>>(in, out, n);

  cudaMemcpy(ou_h, out, sizeof(float)*n, cudaMemcpyDeviceToHost);
  printf("Value of c[2056] is %f\n", ou_h[2056]);
  std::vector<float> in_v(in_h, in_h + n);
  std::vector<float> ou_v(ou_h, ou_h + n);
  double sum_in = std::accumulate(in_v.begin(), in_v.end(), 0.0);
  double sum_ou = std::accumulate(ou_v.begin(), ou_v.end(), 0.0);
  printf("sum_in: %f \nsum_ou: %f\n", sum_in, sum_ou);
  assert(sum_in == sum_ou);
  //
  cudaFree(in);
  cudaFree(out);
  free(in_h);
  free(ou_h);

}
