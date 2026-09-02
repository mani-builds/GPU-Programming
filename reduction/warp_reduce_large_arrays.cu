#include <stdio.h>

#define BLOCK_SIZE 512

__global__ void warp_reduce_kernel(float *in, float *out, int N) {

  // Map data to threads
  int i = 2*blockIdx.x * blockIdx.x + threadIdx.x; // Skip each other block
  // smem Memory layout
  __shared__ float in_s[32]; // Warp size // We only need 32 slots for partial warp sums

  float sum = 0;

  if (i < N) sum += in[i];
  if (i + BLOCK_SIZE < N){
  sum += in[i + BLOCK_SIZE]; // Add two elements (the other from the next block)
  }

  // Logic
  // warp-level primitives for better speed and efficiency

  //1. Reduction across the warps
  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  for (int offset = 16; offset >=1; offset /= 2){
  sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }
  // 2. Store warp sums in shared memory
  if (laneId == 0){
  in_s[warpId] = sum;
  }
  __syncthreads(); // wait for all the warps to finish

  // 3. Final Reduction (using the first warp)
  // Only the first warp needs to work now
  if (warpId == 0) {
    // Read from shared memory; if threadIdx.x < num_warps, else use 0
    float val = (threadIdx.x < (blockDim.x / 32)) ? in_s[laneId] : 0;

    for (int offset = 16; offset >= 1; offset /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF,val, offset);
    }
    // HBM
    if (laneId == 0) {
      // *out = val;
      // For large arrays
      atomicAdd(out, val);
    }
  }
}


int main() {
  float *in_h;
  float *out_h;

  int N = 4096;

  in_h = (float *) malloc(N * sizeof(float));
  out_h = (float *) malloc(sizeof(float));

  for(int i=0; i<N; i++) in_h[i] = 1;

  float *in;
  float *out;

  cudaMalloc(&in, N*sizeof(float));
  cudaMalloc(&out, sizeof(float));

  cudaMemcpy(in, in_h, N*sizeof(float), cudaMemcpyHostToDevice);

  int threads = BLOCK_SIZE;
    // Each block processes 2 * BLOCK_SIZE elements
  int blocksPerGrid = (N + (2*threads) - 1)/ (2*threads);
  warp_reduce_kernel<<<blocksPerGrid, threads>>>(in, out, N);

  printf("blocks: %d, threads: %d\n", blocksPerGrid, threads);
  cudaMemcpy(out_h, out, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Output: %f \n", *out_h);

  free(in_h);
  cudaFree(in);
  cudaFree(out);

  return 0;
}
