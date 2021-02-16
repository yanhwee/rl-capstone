from itertools import zip_longest
from collections import deque

def variable_length_mean(*iterables):
    means = deque()
    for values in zip_longest(*iterables, fillvalue=None):
        acc = 0
        count = 0
        for value in values:
            if value is not None:
                acc += value
                count += 1
        mean = acc / count
        means.append(mean)
    return means