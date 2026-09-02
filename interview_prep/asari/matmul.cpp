#include <cstdint>
#include <iostream>
#include <cstdlib>

#define BLOCK 32

void matmul_block(float *A, float *B, float *C, int M, int N, int K) {
  // C[M x N] = A[M x K] * B[K x N]
  for (int i0 = 0; i0 < M; i0 += BLOCK) {
    for (int j0 = 0; j0 < N; j0 += BLOCK) {
      for (int k0 = 0; k0 < K; k0 += BLOCK) {
        // Block multiplication
        for (int i = i0; i < i0 + BLOCK && i < M; ++i) {
          for (int j = j0; j < j0 + BLOCK && j < N; ++j) {
            float sum = 0.0f;
            for (int k = k0; k < k0 + BLOCK && k < K; ++k) {
              sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += sum;
          }
        }
      }
    }
  }
  // Naive
  // C[M x N] = A[M x K] * B[K x N]
  // float acc;
  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < N; j++) {
  //     acc = 0.0;
  //     for (int k = 0; k < K; k++) {
  //      acc +=  A[i * K + k] * B[k * N + j];
  //     }
  //     C[i*N + j] = acc;
  //   }
  // }

}
