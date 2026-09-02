#include <__clang_cuda_builtin_vars.h>
#include <iostream>
#include <cstdio>

using namespace std;

__global__ void SpMV_ELL(ELLMatrix ellMatrix, float *x, float *y) {

  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < ellMatrix.numRows) {
    float sum = 0.0f;
    for (int t = 0; t < ellMatrix.nnzPerRow[row]; t++) {
      int i = t * ellMatrix.numRows + row;
      float value = ellMatrix.value[i];
      int col = ellMatrix.colIdx[i];

      sum += x[col] * value;
    }
    //HBM
    y[row] = sum;
    
  }

}
