from v2.utils import meshstack_apply
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def batch_eps_rewards(batch_eps_rewards):
    for eps_rewards in batch_eps_rewards:
        xs = range(1, len(eps_rewards) + 1)
        plt.plot(xs, eps_rewards)
    plt.ylabel('Total Reward')
    plt.xlabel('Episodes')
    plt.show()

def eps_rewards(eps_rewards):
    batch_eps_rewards([eps_rewards])

def histories(hists):
    has_key = lambda key: key in hists[0]
    get_values = lambda key: [hist[key] for hist in hists]
    if has_key('eps_rewards'):
        batch_eps_rewards(get_values('eps_rewards'))

def history(hist):
    histories([hist])

def surface_3d(
    zf, xs, ys, xlabel=None, ylabel=None, zlabel=None,
    title=None, colorbar=None, invert_z=None):
    if colorbar is None: colorbar = False
    if invert_z is None: invert_z = False
    xs, ys = np.meshgrid(xs, ys)
    xys = np.dstack((xs, ys))
    zs = np.apply_along_axis(zf, 2, xys)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        xs, ys, zs, cmap=cm.coolwarm, # pylint: disable=maybe-no-member
        linewidth=0, antialiased=False)
    if colorbar: fig.colorbar(surf, shrink=0.5, aspect=5)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if zlabel: ax.set_zlabel(zlabel)
    if title: ax.set_title(title)
    if invert_z: ax.set_zlim(ax.get_zlim()[::-1])
    plt.show()

def heatmap_2d(
    zf, xs, ys, xlabel=None, ylabel=None, title=None,
    linewidths=None, annot=None, ticks=None):
    if linewidths is None: linewidths = 0
    if annot is None: annot = False
    if ticks is None: ticks = False
    xticks = xs if ticks else False
    yticks = ys if ticks else False
    zs = meshstack_apply(zf, xs, ys)
    ax = sns.heatmap(
        zs, linewidths=linewidths, annot=annot,
        xticklabels=xticks, yticklabels=yticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def value_function_2d(
    value_function, lows, highs, intervals=25,
    xlabel=None, ylabel=None, zlabel=None,
    title=None, colorbar=None, invert_z=None,
    heatmap=None, linewidths=None, annot=None):
    assert(len(lows) == len(highs) == 2)
    if heatmap is None: heatmap = False
    linspace = lambda low, high: np.linspace(
        low, high, intervals, dtype=np.float32)
    xs = linspace(lows[0], highs[0])
    ys = linspace(lows[1], highs[1])
    if heatmap:
        heatmap_2d(
            zf=value_function, xs=xs, ys=ys,
            xlabel=xlabel, ylabel=ylabel, title=title,
            linewidths=linewidths, annot=annot)
    else:
        surface_3d(
            zf=value_function, xs=xs, ys=ys,
            xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
            title=title, colorbar=colorbar, invert_z=invert_z)