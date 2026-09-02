#include <stdio.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

// Can be used with more than just 1 Block (1024 thread limitaion), unlike others.

__global__ void CoarsenedSumReductionKernel(float *input, float *output) {

  __shared__ float input_s[BLOCK_DIM];
  unsigned int segment = 2 * blockDim.x * blockIdx.x * COARSE_FACTOR;
  unsigned int i =  threadIdx.x + segment;
  unsigned int t = threadIdx.x;
  float sum = input[i];
  for (int tile = 1; tile < COARSE_FACTOR*2; tile++) {
    // Threads act independtly so no calls to __syncthreas();
    sum += input[i+tile*BLOCK_DIM];
  }
  input_s[t] = sum;
  for (int stride = blockDim.x/2; stride >= 1; stride /= 2) { // 1st iteration is already done, so blockDim.x/2
    __syncthreads(); // move synchronize to the start of 'for' loop, so that the initial
                     // load from Global to Shared Mem happens completely
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}

int main() {

  float *input_h;
  float *output_h;
  float init_value = 0.0f;
  output_h = &init_value;

  int N = 4*1024;

  printf("Intial value: %f \n", *output_h);
  input_h = (float *)malloc(N * sizeof(float));
  // memset(input_h, 1.0, N*sizeof(float));
  for(int i=0; i <N; i++){ input_h[i]=1.0f;}
  float *input;
  float *output;

  cudaMalloc(&input, N * sizeof(float));
  cudaMemcpy(input, input_h, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&output, sizeof(float));
  int threads = BLOCK_DIM;
  int blocks = int((N + BLOCK_DIM -1)/BLOCK_DIM)/2/COARSE_FACTOR;
  printf("threads: %d, blocks: %d\n", threads, blocks);
  CoarsenedSumReductionKernel<<<blocks, threads>>>(input,output);

  cudaMemcpy(output_h, output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("Each thread process two data elements\n");
  printf("FInal value: %f\n", *output_h);
  cudaFree(input);
  cudaFree(output);
  free(input_h);
  return 0;
}
