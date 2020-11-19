import operator as op
from functools import reduce
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt

def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def prod(iterable):
    return reduce(op.mul, iterable, 1)

def key_map(xs):
    return np.array(list(accumulate(xs, op.mul))) // xs[0]

def compose(*fs):
    compose2 = lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs))
    return reduce(compose2, fs)

def simple_plot(ys, x1, ylabel, xlabel, title=None, plot_type=None):
    xs = range(x1, len(ys) + x1)
    if plot_type == 'line':  plt.plot(xs, ys)
    elif plot_type == 'bar': plt.bar(xs, ys)
    else: raise Exception('Plot type is invalid')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(xs)
    if title is not None:
        plt.title(title)
    plt.show()

def simple_line(ys, x1, ylabel, xlabel, title=None):
    simple_plot(ys, x1, ylabel, xlabel, title, 'line')

def simple_bar(ys, x1, ylabel, xlabel, title=None):
    simple_plot(ys, x1, ylabel, xlabel, title, 'bar')