# from itertools import islice

# def reverse_tail(n, iterable):
#     return islice(reversed(iterable), n)

# def reverse_tail_zip(n, *iterables):
#     return islice(zip(*[])))

def reverse_tail_zip(end, *iterables):
    return list(zip(*iterables))[:end:-1]

from timeit import timeit

setup = \
'''
from collections import deque
from itertools import islice
n = 10
k = 3
iterables = tuple(deque(range(n)) for _ in range(k))
length = n - 1
right = n - 1
left = n - length - 1
right1 = right + 1
left1 = left + 1
#gc.enable()
'''
codes = [
    'list(islice(zip(*[reversed(x) for x in iterables]), length))',
    'list(reversed(list(islice(zip(*iterables), left1, right1))))',
    'list(list(zip(*iterables))[right:left:-1])',
    # 'list(zip(*[islice(reversed(iterable), length) for iterable in iterables]))',
    # 'list(zip(*(islice(reversed(iterable), length) for iterable in iterables)))',
    # 'list(islice(zip(*(reversed(iterable) for iterable in iterables)), length))',
    'list(islice(list(zip(*iterables))[::-1], length))',
    'list(reversed(deque(zip(*iterables), maxlen=length)))',
]
# for code in codes:
#     exec(f'{setup}print({code})')
for i, code in enumerate(codes):
    print(i, timeit(code, setup))

timings = [0] * len(codes)
for i in range(1000):
    for j, code in enumerate(codes):
        timings[j] += timeit(code, setup, number=1000)
for i, timing in enumerate(timings):
    print(i, timing)