from itertools import product
from math import pi
import numpy as np

def aggregate(lows, highs, intervals):
    intervals *= np.ones(len(lows), dtype=int)
    highs = np.nextafter(highs, np.inf)
    widths = (highs - lows) / intervals
    keys = np.cumprod(intervals) // intervals[0]
    clipmax = intervals - 1
    return lambda x: np.dot(keys, np.clip(
        ((x - lows) / widths).astype(int), 0, clipmax))

def tab2lin(n_states):
    identity = np.eye(n_states)
    return lambda x: identity[x]

def normalise(lows, highs):
    diff = highs - lows
    return lambda x: x - lows / diff

def fourier(ns):
    c = np.array(list(product(*map(range, ns))))
    return lambda x: np.cos(pi * c @ x)