"""
CONCLUSION:
DEQUE IS STILL BETTER THAN CIRCULAR BUFFER
"""

class CircularBuffer:
    def __init__(self, maxlen):
        self.data = [None] * maxlen
        self.index = 0
    def append(self, value):
        self.data[self.index % len(self.data)] = value
        self.index += 1
    def full(self):
        return self.index >= len(self.data)
    def __len__(self):
        return min(self.index, len(self.data))
    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= 0: key -= len(self)
            if key >= 0 or key < -len(self): raise IndexError
            return self.data[(self.index + key) % len(self.data)]
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            start -= len(self)
            stop -= len(self)
            return [self.data[(self.index + j) % len(self.data)] for j in range(start, stop, step)]
        raise TypeError
    def __setitem__(self, key, value):
        if isinstance(key, int):
            if key >= 0: key -= len(self)
            if key >= 0 or key < -len(self): raise IndexError
            self.data[(self.index + key) % len(self.data)] = value
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            start -= len(self)
            stop -= len(self)
            for j, x in zip(range(start, stop, step), value):
                self.data[(self.index + j) % len(self.data)] = x
    def __repr__(self):
        return f'CircularBuffer({self[:]}, maxlen={len(self.data)})'

append_buffer_setup = \
'''
from __main__ import CircularBuffer
'''
append_buffer_code = \
'''
buffer = CircularBuffer(999)
for i in range(99999):
    buffer.append(1)
'''
append_deque_setup = \
'''
from collections import deque
'''
append_deque_code = \
'''
buffer = deque(maxlen=999)
for i in range(99999):
    buffer.append(1)
'''

read_buffer_setup = \
'''
from __main__ import CircularBuffer
buffer = CircularBuffer(999)
for i in range(999):
    buffer.append(1)
'''
read_buffer_code = \
'''
for i in range(999):
    buffer[i]
'''
read_deque_setup = \
'''
from collections import deque
buffer = deque(maxlen=999)
for i in range(999):
    buffer.append(1)
'''
read_deque_code = \
'''
for i in range(999):
    buffer[i]
'''

from timeit import timeit
print('Deque:', timeit(read_deque_code, read_deque_setup, number=10))
print('Buffer:', timeit(read_buffer_code, read_buffer_setup, number=10))