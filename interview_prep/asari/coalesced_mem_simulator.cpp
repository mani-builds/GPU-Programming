#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <unordered_set>

// Simulate how 32 threads (a warp) access global memory.
// Given an array of 32 target memory addresses requested by 32 threads,
// write an algorithm that calculates how many 128-byte cache lines must be fetched to satisfy the request.

int main() {
  uintptr_t addresses[32] = {/* fill with 32 addresses */};
  size_t cache_line_width = 128;

  std::unordered_set<size_t> cache_lines;
    for (int i = 0; i < 32; ++i) {
        size_t line = addresses[i] / cache_line_width;
        cache_lines.insert(line);
    }
    std::cout << "num of cache lines: " << cache_lines.size() << std::endl;
}
