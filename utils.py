import operator as op
from functools import reduce
from itertools import accumulate, count
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import clear_output
import time
from gym import spaces

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
    return lambda x: int(np.dot(key, x))

def compose(*fs):
    compose2 = lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs))
    return reduce(compose2, fs)

def simple_plot(ys, x1, ylabel, xlabel, title=None, plot_type=None):
    if isinstance(x1, int):
        xs = range(x1, len(ys) + x1)
    else:
        xs = x1
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

def linear_decay_clip(c, y, x):
    assert(y <= c)
    m = (y - c) / x
    return lambda t: max(y, m * t + c)
    
def render_env(env, delay, text):
    clear_output(wait=True)
    env.render()
    print(text)
    time.sleep(delay)

def plot_3d(zf, xs, ys, zlabel=None, xlabel=None, ylabel=None, title=None, invert_z=False, anim=False):
    xs, ys = np.meshgrid(xs, ys)
    xys = np.dstack((xs, ys))
    zs = np.array(zf(xys.reshape(-1, 2))).reshape(xys.shape[:2])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        xs, ys, zs, cmap=cm.coolwarm, 
        linewidth=0, antialiased=False) # pylint: disable=maybe-no-member
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    if zlabel: ax.set_zlabel(zlabel)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if invert_z: ax.set_zlim(ax.get_zlim()[::-1])
    if anim:
        try:
            for angle in count(0, 1):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(0.0001)
        except Exception as e:
            print(e)
    plt.show()

def plot_2d_value_function(vf, env, intervals, labels=None, title=None, invert_v=False, anim=False):
    assert(isinstance(env.observation_space, spaces.Box))
    assert(env.observation_space.shape == (2,))
    lows = env.observation_space.low
    highs = env.observation_space.high
    if labels is None: labels = (None, None, None)
    plot_3d(
        zf=vf,
        xs=np.linspace(lows[0], highs[0], intervals, dtype=np.float32),
        ys=np.linspace(lows[1], highs[1], intervals, dtype=np.float32),
        zlabel=labels[0],
        xlabel=labels[1],
        ylabel=labels[2],
        title=title,
        invert_z=invert_v,
        anim=anim)

def normaliser(lows, highs):
    scale = (highs - lows) / 2
    mean = (lows + highs) / 2
    return lambda x: (x - mean) / scale