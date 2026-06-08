#include <cstdint>
#include <iostream>
#include <vector>
#include <cstdio>

typedef uintptr_t up;

void* align(void* raw_ptr, int alignment) {

  up curr_addr = (up) raw_ptr;
  up aligned_ptr = (curr_addr + (alignment - 1)) & ~(alignment - 1);

  return (void*) aligned_ptr;
}

int main() {
  float *a;
  a = (float *)malloc(sizeof(float));
  void *a1 = align(a,4);
  void *a2 = align(a,8);
  void *a3 = align(a,32);
  void *a4 = align(a,64);
  void *a5 = align(a,128);

  std::cout << "Original ptr: " << (up) a <<std::endl;
  std::cout << "Aligned (4-Byte) ptr: " << (up) a1 <<std::endl;
  std::cout << "Aligned (8-Byte) ptr: " << (up) a2 <<std::endl;
  std::cout << "Aligned (32-Byte) ptr: " << (up) a3 <<std::endl;
  std::cout << "Aligned (64-Byte) ptr: " << (up) a4 <<std::endl;
  std::cout << "Aligned (128-Byte) ptr: " << (up) a5 <<std::endl;
}
