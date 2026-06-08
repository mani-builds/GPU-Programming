#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <vector>

// Float32 to Int8
// Ref: https://en.wikipedia.org/wiki/Single-precision_floating-point_format for f32 to binary

typedef float f32;
typedef int8_t i8;
typedef int32_t i32;

std::vector<bool> binary(int k) {
  std::vector<bool>bin;
  int rem;

  while (k != 0 ) {
    rem = k % 2;
    bin.insert(bin.begin(), rem);
    k = k / 2;
  }
  return bin;
}

std::vector<bool> fractional_binary(float k) {
  std::vector<bool>bin;
  float f;
  int s;
  if (k == 0.0) {
    bin.push_back(0);
    return bin;
  }
  while(k != 0.0){
  f = k * 2;
  s = (int) f;
  bin.push_back(s);
  k = f - (float) s;
  }
  return bin;
}

std::vector<bool> float_to_bits(float a) {
  std::vector<bool> bin(32, 0);
  std::vector<bool> int_bits;
  std::vector<bool> fractional_bits;

  // First bit (sign)
  bin[0] = (a < 0) ? 1 : 0;

  int int_part = (int) a;
  float fractional_part = a - (float) int_part;
  int_bits = binary(int_part);
  fractional_bits = fractional_binary(fractional_part);

  int e = int_bits.size() - 1; //exponent or shift

  //exponent (8-bits)
  e = 127 + e;
  int tmp = 1;
  for (bool i : binary(e)) {
    bin[tmp] = i;
    tmp++;
  }
  //fraction (23-bits)
  tmp = 9;
  int_bits.erase(int_bits.begin());
  for (bool j : int_bits) {
    bin[tmp] = j;
    tmp++;
  }
  for (bool j : fractional_bits) {
    bin[tmp] = j;
    tmp++;
  }
  return bin;
}

int main() {
  f32 k = 12.375;
  i8 v;
  i32 v1;
  v = (i8) k;
  v1 = k;

 std::vector<bool>bin =  binary(12);
  for (int i : bin) std::cout << i <<  " ";
  // std::cout << "Float32: " << k  << " "<< "Int32: " << v1 << std::endl;
  //
std::vector<bool>frac_bin =  fractional_binary(0.375);

  std::cout << "\nFractional bin: " << std::endl;
  for (int i : frac_bin) std::cout << i <<  " ";

std::vector<bool>frac_bits =  float_to_bits(0.375);
  std::cout << "\nFractional bin: " << std::endl;
  for (int i : frac_bits) std::cout << i <<  " ";



}
