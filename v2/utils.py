from random import random
from functools import reduce
import numpy as np

###  Misc  ###
def randbool(p):
    return random() < p

def argmax(values):
    return np.argmax(values)

def do_nothing(*args, **kwargs):
    pass

def array_walrus(array, index, value):
    array[index] = value
    return array

def compose(*fs):
    compose2 = lambda f, g: \
        lambda *args, **kwargs: g(f(*args, **kwargs))
    return reduce(compose2, fs)

def meshstack(*xs):
    return np.dstack(np.meshgrid(*xs))

def meshstack_apply(f, *xs):
    return np.apply_along_axis(f, 2, meshstack(*xs))

###  Schedules  ###
def linear_decay_clip(c, y, x):
    assert(y <= c)
    m = (y - c) / x
    return lambda ts: max(y, m * ts)