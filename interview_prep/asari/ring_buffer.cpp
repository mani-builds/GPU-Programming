#include <iostream>
#include <vector>

template<typename T>
class RingBuffer {
public:
    RingBuffer(size_t capacity)
        : buffer(capacity), head(0), tail(0), full(false) {}

    void push(const T& item) {
        buffer[tail] = item;
        if (full) {
            head = (head + 1) % buffer.size();
        }
        tail = (tail + 1) % buffer.size();
        full = tail == head;
    }

    T pop() {
        if (empty()) {
            throw std::runtime_error("Buffer is empty");
        }
        T item = buffer[head];
        full = false;
        head = (head + 1) % buffer.size();
        return item;
    }

    bool empty() const {
        return (!full && (head == tail));
    }

    bool is_full() const {
        return full;
    }

    size_t size() const {
        if (full) {
            return buffer.size();
        }
        if (tail >= head) {
            return tail - head;
        }
        return buffer.size() - head + tail;
    }

private:
    std::vector<T> buffer;
    size_t head;
    size_t tail;
    bool full;
};

// Example usage
int main() {
    RingBuffer<int> rb(3);
    rb.push(1);
    rb.push(2);
    rb.push(3);
    std::cout << rb.pop() << std::endl; // 1
    rb.push(4);
    std::cout << rb.pop() << std::endl; // 2
    std::cout << rb.pop() << std::endl; // 3
    std::cout << rb.pop() << std::endl; // 4
    return 0;
}
