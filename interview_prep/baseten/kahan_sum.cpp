
float kahan_sum(const float *arr, int N) {
  float sum = 0.0f;
  float c = 0.0f;
  for (int i = 0; i < N; i++) {
    float y = arr[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}
