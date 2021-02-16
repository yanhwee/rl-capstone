import v2.linear as Linear
from v2.features import fourier
from math import pi, sin, cos, exp
from random import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fs = {
        'Linear': lambda x: x,
        'Half Step': lambda x: 0 if x < 0.5 else 1,
        'Sine': lambda x: sin(2 * pi * x),
        'Cosine': lambda x: cos(4 * pi * x),
        'Exponential': lambda x: exp(x)
    }

    n = 20
    p = fourier([n])
    weights = Linear.weights(n, 1)
    def update(x, y, lr=0.01):
        Linear.update_q_target(
            weights, lr, p([x]), 0, y)
    v = lambda x: Linear.qf(weights)(p([x]), 0)
    
    iterations = 100
    batch_size = 10
    xs = np.linspace(0, 1, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    l1, l2 = ax.plot(xs, xs, xs, xs)
    ax.autoscale_view(True, True, True)
    plt.draw()
    plt.pause(1e-9)
    input()

    for name, f in fs.items():
        l1.set_ydata(np.vectorize(f)(xs))
        l2.set_ydata(l1.get_ydata())
        ax.set_title(f'Fourier ({n} terms) ({name})')
        ax.relim()
        ax.autoscale_view()
        for i in range(iterations):
            for j in range(batch_size):
                x = random()
                y = f(x)
                update(x, y)
            l2.set_ydata(np.vectorize(v)(xs))
            plt.draw()
            plt.pause(1e-9)