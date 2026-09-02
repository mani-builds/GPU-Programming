#include <iostream>
#include <cstdio>

using namespace std;

#define BLOCK 128

typedef struct {
    int numRows;
    int numCols;
    int numNonzeros;
    int *rowIdx;
    int *colIdx;
    float *value;
} COOMatrix;

__global__ void SpMV(COOMatrix cooMatrix, float *x, float *y) {
  // Map each 'Value' data to thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // MatV mul
  if (i < cooMatrix.numNonzeros) {
    int row = cooMatrix.rowIdx[i];
    int col = cooMatrix.colIdx[i];
    float value = cooMatrix.value[i];
    // HBM
    atomicAdd(&y[row], x[col] * value);
    // atomic operations are the trade-off we are willing to make for sparsity
  }

}

int main() {
    // Example COO matrix: 3x3 with 4 nonzeros
    int numRows = 3, numCols = 3, numNonzeros = 4;
    int h_rowIdx[] = {0, 0, 1, 2};
    int h_colIdx[] = {0, 2, 1, 2};
    float h_value[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float h_x[] = {1.0f, 2.0f, 3.0f};
    float h_y[] = {0.0f, 0.0f, 0.0f};

    // Device memory
    int *d_rowIdx, *d_colIdx;
    float *d_value, *d_x, *d_y;

    cudaMalloc(&d_rowIdx, numNonzeros * sizeof(int));
    cudaMalloc(&d_colIdx, numNonzeros * sizeof(int));
    cudaMalloc(&d_value, numNonzeros * sizeof(float));
    cudaMalloc(&d_x, numCols * sizeof(float));
    cudaMalloc(&d_y, numRows * sizeof(float));

    cudaMemcpy(d_rowIdx, h_rowIdx, numNonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, numNonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, numNonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, numRows * sizeof(float), cudaMemcpyHostToDevice);

    COOMatrix d_cooMatrix;
    d_cooMatrix.numRows = numRows;
    d_cooMatrix.numCols = numCols;
    d_cooMatrix.numNonzeros = numNonzeros;
    d_cooMatrix.rowIdx = d_rowIdx;
    d_cooMatrix.colIdx = d_colIdx;
    d_cooMatrix.value = d_value;

    int threadsPerBlock = BLOCK;
    int blocksPerGrid = (numNonzeros + threadsPerBlock - 1) / threadsPerBlock;

    SpMV<<<blocksPerGrid, threadsPerBlock>>>(d_cooMatrix, d_x, d_y);

    cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < numRows; ++i) {
        printf("y[%d] = %f\n", i, h_y[i]);
    }

    // Cleanup
    cudaFree(d_rowIdx);
    cudaFree(d_colIdx);
    cudaFree(d_value);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
