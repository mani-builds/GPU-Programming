#include <cstdint>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <vector>

// Float32 to Int8
// Ref: https://en.wikipedia.org/wiki/Single-precision_floating-point_format for f32 to binary

typedef float f32;
typedef int8_t i8;
typedef int32_t i32;
typedef uint32_t u32;

std::vector<bool> float_to_bits(float a) {
  u32 bits;
  std::memcpy(&bits, &a, sizeof(float)); // Re-interpret float as an integer
  std::vector<bool> bin(32);
  for (int i = 31; i >= 0; i--) {
    bin[31-i] = (bits >> i) & 1;
  }
  return bin;
}

int main() {
  f32 k = 12.375;
  i8 v;
  i32 v1;
  v = (i8) k;
  v1 = k;

std::vector<bool>frac_bits =  float_to_bits(12.375);
  std::cout << "\nFractional bin: " << std::endl;
  for (int i : frac_bits) std::cout << i <<  " ";



}
