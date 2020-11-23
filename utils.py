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

def key_mapper(xs):
    key = key_map(xs)
    return lambda x: np.dot(key, x)

def compose(*fs):
    compose2 = lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs))
    return reduce(compose2, fs)

def simple_plot(ys, x1, ylabel, xlabel, title=None, plot_type=None):
    xs = range(x1, len(ys) + x1)
    if plot_type == 'line':  plt.plot(xs, ys)
    elif plot_type == 'bar': plt.bar(xs, ys)
    elif plot_type == 'scatter': plt.scatter(xs, ys, s=0.1)
    else: raise Exception('Plot type is invalid')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if len(xs) < 30: plt.xticks(xs)
    if title is not None:
        plt.title(title)
    plt.show()

def simple_line(ys, x1, ylabel, xlabel, title=None):
    simple_plot(ys, x1, ylabel, xlabel, title, 'line')

def simple_bar(ys, x1, ylabel, xlabel, title=None):
    simple_plot(ys, x1, ylabel, xlabel, title, 'bar')

def simple_scatter(ys, x1, ylabel, xlabel, title=None):
    simple_plot(ys, x1, ylabel, xlabel, title, 'scatter')

def mae(a, b):
    return np.sum(np.abs(a - b)) / prod(a.shape)

def argmax_equal_count(this, that):
    return np.sum(np.isclose(that[range(this.shape[0]),np.argmax(this,axis=1)], np.amax(that, axis=1)))

def argmax_equal_percent(this, that):
    return argmax_equal_count(this, that) / this.shape[0]