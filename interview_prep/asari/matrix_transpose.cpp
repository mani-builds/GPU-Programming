#include <cstdint>
#include <iostream>
#include <cstdlib>

#define BLOCK 32

void matrix_t(float *mat, float *out, int M, int N) {

  for (int row_ph = 0; row_ph < (M + BLOCK - 1) / BLOCK; row_ph++) {
    for (int col_ph = 0; col_ph < (N + BLOCK - 1) / BLOCK; col_ph++) {
      // Read the block of data(cache-friendly as cache likely has the data near the read)
      for (int i = row_ph * BLOCK; i < (row_ph + 1) * BLOCK && i < M; i++) {
        for (int j = col_ph * BLOCK; j < (col_ph + 1) * BLOCK && j < N; j++) {
          out[j * M + i] = mat[i * N + j];
        }
      }
    }
  }

  // Naive
  // for (int row = 0; row < M; row++) {
  //   for (int col = 0; col < N; col++) {
  //     out[col*M + row] = mat[row * N + col];
  //   }
  // }
}

int main() {
  float *mat, *out;
  int M = 1000;
  int N = 1000;

  srand(time(0));
  mat = (float *) malloc(sizeof(float) * M * N);
  out = (float *) malloc(sizeof(float) * M * N);

  for (int i = 0; i < M * N; i++) {
    mat[i] = rand() % 10;
  }


  std::cout << "Input mat: " <<std::endl;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cout << mat[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
  matrix_t(mat, out, M, N);
std::cout << "Transpoed mat: " <<std::endl;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cout << out[i * M + j] << " ";
    }
    std::cout << std::endl;
  }
}
