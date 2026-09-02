#include <iostream>
#include <cstdio>

using namespace std;

#define NUM_BINS 7

__global__ void parallel_hist(char *a, int *hist, int N) {

  // Map threads to data indices
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  //smem
  __shared__ int hist_s[NUM_BINS];
  for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    // the for-loop helps initialize only NUM_BINS of hist_s
    // It also helps when NUM_BINS > blockDim.x
    hist_s[bin] = 0;
  }
  __syncthreads();

  // Histogram
  if (i < N){
    int alphabet_position = a[i] - 'a';
    if (alphabet_position >=0 && alphabet_position < 26){
        atomicAdd(&(hist_s[alphabet_position/4]), 1);
    }
  }
  __syncthreads();

  // HBM
  for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    int binValue = hist_s[bin];
    if(binValue > 0)
    atomicAdd(&(hist[bin]), hist_s[bin]);
  }
}


int main() {
    const int N = 4096;
    // const int NUM_BINS = 26 / 4 + ((26 % 4) ? 1 : 0); // 7 bins for 26 letters

    // Host allocations
    char a_h[N + 1];
    srand(time(NULL)); // Seed random number generator
    int hist_h[NUM_BINS] = {0};
    for (int i = 0; i < N; ++i) {
        a_h[i] = 'a' + rand() % 26; // Random lowercase letter
    }
    a_h[N] = '\0'; // Null-terminate

    // Device allocations
    char *a_d;
    int *hist_d;
    cudaMalloc(&a_d, sizeof(char) * N);
    cudaMalloc(&hist_d, sizeof(int) * NUM_BINS);

    // Initialize device histogram to zero
    cudaMemset(hist_d, 0, sizeof(int) * NUM_BINS);

    // Copy input to device
    cudaMemcpy(a_d, a_h, sizeof(char) * N, cudaMemcpyHostToDevice);

    // Kernel launch
    int threads = 32;
    int blocks = (N + threads - 1) / threads;
    parallel_hist<<<blocks, threads>>>(a_d, hist_d, N);

    // Copy result back to host
    cudaMemcpy(hist_h, hist_d, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost);

    // Print histogram
    printf("Histogram bins (each bin = 4 letters):\n");
    for (int i = 0; i < NUM_BINS; ++i) {
        printf("%d,", hist_h[i]);
    }
  printf("\n");

    // Free memory
    cudaFree(a_d);
    cudaFree(hist_d);

    return 0;
}
