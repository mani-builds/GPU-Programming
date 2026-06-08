#include <cstdio>
#include <cstdlib>
#include <runner.cuh>
#include <iostream>
#include <ctime>
#include <string>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // device info
  // int deviceId = 0;
  CudaDeviceInfo();

  long m, n, k, max_size;

  max_size = 4096;
  m = n = k = max_size;

  float alpha = 0.5, beta = 3.0;

  float *A_h = nullptr, *B_h = nullptr, *C_h = nullptr; // Host ptrs
  float *A = nullptr, *B = nullptr, *C = nullptr; // Device ptrs

  // Allocate memory
  A_h = (float *) malloc(sizeof(float) * m * k);
  if (A_h == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    return EXIT_FAILURE;
  }
  B_h = (float *) malloc(sizeof(float) * k * n);
  if (B_h == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    return EXIT_FAILURE;
  }
  C_h = (float *) malloc(sizeof(float) * m * n);
  if (C_h == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    return EXIT_FAILURE;
  }

  // Assign values to matrices
  // randomize_matrix(A_h, max_size * max_size);
  // randomize_matrix(B_h, max_size * max_size);
  // randomize_matrix(C_h, max_size * max_size);

  for (int i = 0; i < max_size * max_size; i++) {
    A_h[i] = 1.0;
    B_h[i] = 1.0;
    C_h[i] = 1.0;
  }

  printf("Intial matrix C (25-values): \n");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%f ", C_h[i*m + j]);
    }
    printf("\n");
  }

  // Copy data from host to device
  cudaCheck(cudaMalloc(&A, sizeof(float) * m * k));
  cudaCheck(cudaMalloc(&B, sizeof(float) * k * n));
  cudaCheck(cudaMalloc(&C, sizeof(float) * m * n));

  cudaCheck(cudaMemcpy(A, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(B, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(C, C_h, sizeof(float) * m * n, cudaMemcpyHostToDevice));

  std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
  run_kernel(kernel_num, m, n, k, alpha, A, B, beta, C); // Executes the kernel, modifies the result matrix

  cudaCheck(cudaDeviceSynchronize());
  cudaMemcpy(C_h, C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  printf("Final matrix C (25-values): \n");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%f ", C_h[i*m + j]);
    }
    printf("\n");
  }

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}
