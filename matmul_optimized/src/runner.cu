#include "kernel.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/select.h>


void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

void CudaDeviceInfo() {

  int devCount;

  cudaGetDeviceCount(&devCount);

  cudaDeviceProp props{}; // brace-initialization, props = 0

  for (int i = 0; i < devCount; i++) {
    cudaGetDeviceProperties(&props, i);
    printf("Device properties for Device ID : %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         i, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
  }
}

void randomize_matrix(float *mat, int N) {
  struct timeval time{};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp *(-1.);
    mat[i] = tmp;
  }
}

void copy_matrix(float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 blockDim(32, 32);
  dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

}

void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  dim3 blockDim(32, 32);
  dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
  sgemm_global_mem_coalesce
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 blockDim(32, 32);
  dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  cudaFuncSetAttribute(sgemm_shared_mem_block,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C) {
  switch (kernel_num) {
  case 0:
    // runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  // case 4:
  //   runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 5:
  //   runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 6:
  //   runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 7:
  //   runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 8:
  //   runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 9:
  //   runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 10:
  //   runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 11:
  //   runSgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 12:
  //   runSgemmDoubleBuffering2(M, N, K, alpha, A, B, beta, C);
  //   break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }

}
