#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>


typedef uintptr_t up;

#define ARENA_ALIGN (sizeof(void*))
#define ARENA_UP_POW2(n, p) (((uint64_t)(n) + ((uint64_t)(p) - 1)) & (~(uint64_t)(p)-1))

int storage_left = 100;
void *base_ptr;
uintptr_t next_addr;

void* arena_alloc() {
  base_ptr = malloc(100); // allocate 100 bytes of memory
  next_addr = (up) base_ptr;
  return base_ptr;
}

void arena_free(void *ptr) {
  if (ptr != NULL) {
    free(ptr);
  }
}

void *block_alloc(size_t size) {
  if (storage_left - (int) size < 0) {
    printf("Not enough storage\n");
    return NULL;
  }
  up current_addr = next_addr;
  // if (next_addr == (uintptr_t) base_ptr) {
  //   next_addr = next_addr + size;
  //   storage_left = storage_left - size;
  //   return base_ptr;
  // }
  next_addr = next_addr + size;
  storage_left = storage_left - (int) size;
  return (void *) current_addr;
}

int main() {
  void *start_ptr;
  start_ptr = arena_alloc();

  std::cout << "Initial ptr: " << start_ptr << "," << (uintptr_t)start_ptr << std::endl;
  // To perform numerics
  // next_addr = (uintptr_t) start_ptr;

  // 4 floats = 16 bytes
  float *a;
  a = (float *) block_alloc(4 * sizeof(float));
  std::cout << "Size "<< 4 * sizeof(float) << std::endl;
  // a = (float *)next_addr;
  std::cout << "Float ptr: " << a << "," << (uintptr_t)a << std::endl;

  // 8 ints = 32 bytes
  int *i;
  // i = (int *)next_addr;
  i = (int *) block_alloc(8 * sizeof(int));
  std::cout << "Int ptr: " << i << "," << (uintptr_t)i << std::endl;


  // 8 ints = 32 bytes
  int *i2;
  i2 = (int *) block_alloc(8 * sizeof(int));
  std::cout << "Int2 ptr: " << i2 << "," << (uintptr_t)i2 << std::endl;


  std::cout << "Storage left "<< storage_left << std::endl;
  // 8 ints = 32 bytes
  int *i3;
  i3 = (int *) block_alloc(8 * sizeof(int));
  std::cout << "Int3 ptr: " << i3 << "," << (uintptr_t)i3 << std::endl;

  std::cout << "Storage left "<< storage_left << std::endl;

  arena_free(start_ptr);
}
