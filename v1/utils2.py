import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import clear_output

###  Schedules  ###
def linear_decay_clip(c, y, x):
    assert(y <= c)
    m = (y - c) / x 
    return lambda t: max(y, m * t)

###  Gym  ###
def render_env(env, delay=0, text=''):
    clear_output(wait=True)
    env.render()
    print(text)
    time.sleep(delay)

def test_env(env, delay=0, ts=int(1e9), action_func=None):
    if action_func is None:
        action_func = lambda state: env.action_space.sample()
    acc_reward = 0
    state = env.reset()
    render_env(env, delay, 0)
    for t in range(ts):
        action = action_func(state)
        state, reward, done, _ = env.step(action)
        acc_reward += reward
        render_env(env, delay, t)
        if done: break
    env.close()

###  Plot  ###
def plot_3d(
    zf, xs, ys, zlabel=None, xlabel=None, ylabel=None, 
    title=None, colorbar=False, invert_z=False):
    xs, ys = np.meshgrid(xs, ys)
    xys = np.dstack((xs, ys))
    zs = np.apply_along_axis(zf, 2, xys)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        xs, ys, zs, cmap=cm.coolwarm, # pylint: disable=maybe-no-member
        linewidth=0, antialiased=False) 
    if colorbar: fig.colorbar(surf, shrink=0.5, aspect=5)
    if zlabel: ax.set_zlabel(zlabel)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if invert_z: ax.set_zlim(ax.get_zlim()[::-1])
    plt.show()

def plot_2d_value_function(
    value_function, lows, highs, intervals=25,
    zlabel=None, xlabel=None, ylabel=None, 
    title=None, colorbar=False, invert_z=False):
    assert(len(lows) == len(highs) == 2)
    linspace = lambda low, high: np.linspace(
        low, high, intervals, dtype=np.float32)
    plot_3d(
        zf=value_function,
        xs=linspace(lows[0], highs[0]),
        ys=linspace(lows[1], highs[1]),
        zlabel=zlabel, xlabel=xlabel, ylabel=ylabel, title=title,
        invert_z=invert_z)