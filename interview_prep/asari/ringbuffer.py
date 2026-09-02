#!/usr/bin/env python

class RingBuffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.head = 0
        self.tail = 0
        self.full = False

    def push(self, item):
        self.buffer[self.tail] = item
        if self.full:
            self.head = (self.head + 1) % self.capacity
        self.tail = (self.tail + 1) % self.capacity
        self.full = self.tail == self.head

    def pop(self):
        if self.empty():
            raise IndexError("Buffer is empty")
        item = self.buffer[self.head]
        self.full = False
        self.head = (self.head + 1) % self.capacity
        return item

    def empty(self):
        return (not self.full) and (self.head == self.tail)

    def is_full(self):
        return self.full

    def size(self):
        if self.full:
            return self.capacity
        if self.tail >= self.head:
            return self.tail - self.head
        return self.capacity - self.head + self.tail
