#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>

class Dict{
  public:
  int n;
  std::vector<int> key; // 4-bytes
  std::vector<char> val; // 1-byte
  int count = 0;
  Dict(int i){ //: n(i) {}
    n = i;
    key.reserve(n);
    val.reserve(n);
  }

  void put(int k, char v) {
        // Check if key exists
        auto it = std::find(key.begin(), key.end(), k);
        if (it != key.end()) {
            // Remove existing key and value
            int idx = it - key.begin();
            key.erase(key.begin() + idx);
            val.erase(val.begin() + idx);
            count--;
        }
        // Insert at front
        key.insert(key.begin(), k);
        val.insert(val.begin(), v);
        count++;
        // If over capacity, remove last
        if (count > n) {
            del_last();
        }
    }

  void del_last(){
    if (!key.empty() && !val.empty()) {
            key.pop_back();
            val.pop_back();
            count--;
    }
  }

  void print() {
        std::cout << "Current LRU state:\n";
        for (int i = 0; i < count; ++i) {
            std::cout << key[i] << ":" << val[i] << " ";
        }
        std::cout << std::endl;
    }

};

void lru(Dict& d) {
    int key;
    char value;
    while (true) {
        std::cout << "Enter the Key int (negative to quit): ";
        std::cin >> key;
        if (key < 0) break;
        std::cout << "Enter the Value char: ";
        std::cin >> value;

        d.put(key, value);
        d.print();
    }
}

int main() {
  int n = 5;
  Dict d(n);

  // int count = 0;
  // for(int i = 0; i < n; i++){
  //   d.put(i, 'A' + i);
  //   count++;
  // }
  //

  lru(d);

  // std::cout << "d[n-1].key: " <<d.key[n-1] << " d[n-1].value: " <<d.val[n-1] << std::endl;
  // std::cout << "d[i].key: " <<d[0].key << " d[i].value: " <<d[0].val << std::endl;


}
