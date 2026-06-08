#include <cmath>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <typeinfo>
#include <cstdint>

struct Employee {
  char initial; // 1-byte
  int id; // 4-bytes
};

// bytes - number of bytes to allocate(in aligned-world), align - alignment boundary
void* aligned_malloc(size_t bytes, size_t alignment) {
  // We need extra space for the alignment padding AND to store the original pointer
  size_t total_bytes = bytes + (alignment-1) + sizeof(void *);

  void* raw_ptr = malloc(total_bytes);

  // Use uintptr_t to perform math
  uintptr_t raw_addr = (uintptr_t) raw_ptr;
  uintptr_t aligned_addr =
      (raw_addr + sizeof(void *) + (alignment - 1)) & ~(alignment - 1);
  // ~(alignEmployeeent - 1) acts as a mark (for 4byte the last two bits will be 00,
  // so every bit-wise-add will result in 00 as last bits, i.e divisible by 4)

  //store original pointer to be used in free();
  // Store the original pointer right before the aligned address
  void** ptr_storage = (void**) aligned_addr;
  ptr_storage[-1] = raw_ptr;

  return (void *) aligned_addr;
}

void aligned_free(void *aligned_addr) {

  void** ptr_storage = (void**) aligned_addr;
  void *raw_ptr = ptr_storage[-1];

  free(raw_ptr);
}

int main() {
  Employee *array;

// Let's try assigning 3 elEmployeeents for array, so total needed = 15 bytes
// If initial elEmployeeent is at addr '0', (1 byte char is padded to 4 bytes and 4 bytes int is 4 bytes)
// next elEmployeeent should be at addr '8',
// next elEmployeeent should be at addr '16',
// (if 4-byte alignment), if malloc returns an addr of '2', the aligned_malloc
// should return a ptr of '4' by adding padding

  int n = 3;

  size_t bytes_with_padding = n * 8;

  array = (Employee *)aligned_malloc(bytes_with_padding, 4);

  std::cout << "Pointer after aligned:array " << array << std::endl;
  std::cout << "Pointer at array[1] " << &array[1] << std::endl;
  std::cout << "Pointer at array[2] " << &array[2] << std::endl;
  std::cout << "Pointer at array[3] " << &array[3] << std::endl;
  std::cout << "Pointer at array[0].initial " << &array[1].initial << std::endl;
  std::cout << "Pointer at array[0].id " << &array[1].id << std::endl;

  aligned_free(array);

  return 0;
}
