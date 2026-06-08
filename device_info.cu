#include <iostream>
#include <cuda_runtime.h>

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
    int driverVersion = 0;
    int runtimeVersion = 0;

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

  printf("---------------------------------------\n");
    std::cout << "Driver Version (max supported): "
              << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << "\n";
    std::cout << "Runtime Version (nvcc compiler): "
              << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << "\n";
}

int main() {
  CudaDeviceInfo();
  return 0;
}
