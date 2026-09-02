#include <iostream>
#include <cstdio>

using namespace std;

#define FILTER_SIZE 4

__global__ void stencil(float *in, float *ou, float *filter, int M, int N) {

  // Map threads to output data
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Logic
  // Edges
  if (col < N && row < M){
    if (col == 0 || row == 0 || col == (N - 1) || row == (M - 1)) {
        ou[row * N + col] = in[row * N + col];
    } else {
      float left = in[row * N + (col - 1)];
      float right = in[row * N + (col + 1)];
      float top = in[(row - 1) * N + col];
      float bottom = in[(row + 1) * N + col];

      // HBM
      ou[row * N + col] = filter[0] * left + filter[1] * right +
                          filter[2] * top + filter[3] * bottom;
    }
  }
}

int main() {

  int M = 10;
  int N = 10;

  float *in_h, *ou_h;
  float filter_h[FILTER_SIZE] = {1,1,1,1};

  in_h = (float *)malloc(sizeof(float) * M * N);
  ou_h = (float *)malloc(sizeof(float) * M * N);

  for (int i = 0; i < M * N; i++) {
    in_h[i] = 1.0f;
    ou_h[i] = 0.0f;
  }

  printf("Input matrix: \n");
  for (int i = 0; i < 10 ; i++) {
    for (int j = 0; j < 10 ; j++) {
      cout << in_h[i*N + j] << " ";
    }
      cout << endl;
  }

  float *in, *ou, *filter;
  cudaMalloc(&in, sizeof(float) * M * N);
  cudaMalloc(&ou, sizeof(float) * M * N);
  cudaMalloc(&filter, sizeof(float) * FILTER_SIZE);

  cudaMemcpy(in, in_h, sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(ou, ou_h, sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(filter, filter_h, sizeof(float)*FILTER_SIZE, cudaMemcpyHostToDevice);

  dim3 threads(16,16);
  dim3 blocks((N + threads.x - 1)/ threads.x, (M + threads.y - 1)/ threads.y);

  stencil<<<blocks, threads>>>(in,ou, filter, M, N);

  cudaMemcpy(ou_h, ou, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  printf("Output matrix: \n");
  for (int i = 0; i < 10 ; i++) {
    for (int j = 0; j < 10 ; j++) {
      cout << ou_h[i*N + j] << " ";
    }
      cout << endl;
  }

  cudaFree(filter);
  cudaFree(in);
  cudaFree(ou);
  free(in_h);
  free(ou_h);

}
