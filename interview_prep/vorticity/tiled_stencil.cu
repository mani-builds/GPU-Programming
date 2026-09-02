#include <iostream>
#include <cstdio>

using namespace std;

// 3-D 7-Point stencil

#define FILTER_SIZE 6

#define RADIUS 2
#define INPUT_TILE 8
#define OUTPUT_TILE ((INPUT_TILE) - (2))


__global__ void tiled_stencil(float *in, float *ou, float *filter, int M, int N, int K) {

  // Map threads to output data
  // Since we the size of block is more than OUTPUT_TILE, we need to subtract to get
  // output's coordinates
  int col = blockIdx.x * OUTPUT_TILE + threadIdx.x - 1;
  int row = blockIdx.y * OUTPUT_TILE + threadIdx.y - 1;
  int width = blockIdx.z * OUTPUT_TILE + threadIdx.z - 1;

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int tz = threadIdx.z;

  // smem
  __shared__ float in_s[INPUT_TILE][INPUT_TILE][INPUT_TILE];
  if (col >= 0 && col < K && row >= 0 && row < N && width >= 0 && width < M) {
    // if-statement is need to gaurd againt halo- or empty-cells
    in_s[tz][ty][tx] = in[width * N * K + row * K + col];
  }
  __syncthreads();

  // Logic
  // Edges
  if (col < K && row < N && width < M){
    if (col == 0 || row == 0 || width == 0 || col == (K - 1) || row == (N - 1) || width == (M -1)) {
        ou[row * N + col] = in_s[width][row][col];
    } else {

      // HBM
      ou[row * N + col] = filter[0] * in_s[tz][ty][tx - 1] +
                          filter[1] * in_s[tz][ty][tx + 1] +
                          filter[2] * in_s[tz][ty - 1][tx] +
                          filter[3] * in_s[tz][ty + 1][tx] +
                          filter[4] * in_s[tz + 1][ty][tx] +
                          filter[5] * in_s[tz - 1][ty][tx];
    }
  }
}

int main() {

  int M = 100;
  int N = 100;
  int K = 100;

  float *in_h, *ou_h;
  float filter_h[FILTER_SIZE] = {1,1,1,1};

  in_h = (float *)malloc(sizeof(float) * M * N * K);
  ou_h = (float *)malloc(sizeof(float) * M * N * K);

  for (int i = 0; i < M * N; i++) {
    in_h[i] = 1.0f;
    ou_h[i] = 0.0f;
  }

  printf("Input matrix: \n");
  for (int i = 0; i < 10 ; i++) {
    for (int j = 0; j < 10 ; j++) {
    for (int k = 0; k < 10 ; j++) {
      cout << in_h[i*M*N + j*N + k] << " ";
    }
    }
      cout << endl;
  }

  float *in, *ou, *filter;
  cudaMalloc(&in, sizeof(float) * M * N * K);
  cudaMalloc(&ou, sizeof(float) * M * N * K);
  cudaMalloc(&filter, sizeof(float) * FILTER_SIZE);

  cudaMemcpy(in, in_h, sizeof(float)*M*N * K, cudaMemcpyHostToDevice);
  cudaMemcpy(ou, ou_h, sizeof(float)*M*N * K, cudaMemcpyHostToDevice);
  cudaMemcpy(filter, filter_h, sizeof(float)*FILTER_SIZE, cudaMemcpyHostToDevice);

  dim3 threads(INPUT_TILE,INPUT_TILE, INPUT_TILE);
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y,
              (K + threads.z - 1)/ threads.z);

  tiled_stencil<<<blocks, threads>>>(in,ou, filter, M, N, K);

  cudaMemcpy(ou_h, ou, sizeof(float)*M*N * K, cudaMemcpyDeviceToHost);

  printf("Output matrix: \n");
  printf("Input matrix: \n");
  for (int i = 0; i < 10 ; i++) {
    for (int j = 0; j < 10 ; j++) {
    for (int k = 0; k < 10 ; j++) {
      cout << ou_h[i*M*N + j*N + k] << " ";
    }
    }
      cout << endl;
  }


  cudaFree(filter);
  cudaFree(in);
  cudaFree(ou);
  free(in_h);
  free(ou_h);

}
