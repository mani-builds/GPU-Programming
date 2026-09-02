#include <__clang_cuda_builtin_vars.h>
#include <iostream>
#include <cstdio>

using namespace std;


__global__ void SpMV_CSR(CSRMatrix csrMatrix, float *x, float*y) {
  // Map out data to threads
  int row = blockDim.x + blockIdx.x + threadIdx.x;

  // Mat-V mul
  if (row < csrMatrix.numRows){
    float sum = 0.0f;
    for (int i = csrMatrix.rowPtrs[row]; row < csrMatrix.rowPtrs[row + 1]; row++) {
      float value = csrMatrix.value[i];
      int col = csrMatrix.colIdx[i];
      sum += value * x[col];
    }
    // HBM
    y[row] += sum;
  }
}

// Launch
int threads = rows;
int blocks = ceil_div(N/rows);
