from v2.env import DiscreteEnv
from v2.sole import sole_step
from v2.dp import dp_step, dp_micro_step
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GridWorld(DiscreteEnv):
    def __init__(self):
        n_rows, n_cols = 5, 5
        n_states = n_rows * n_cols
        n_actions = 4 # Clockwise
        mrc = lambda row, col: row * n_cols + col
        P = {
            state: { action: None for action in range(n_actions) }
            for state in range(n_states) }
        deterministic = lambda state, reward, done: \
            [(1.0, state, reward, done)]
        ok = lambda row, col: 0 <= row < n_rows and 0 <= col < n_cols
        outcome = lambda row, col, fallback: (
            deterministic(mrc(row, col), -1, False)
            if ok(row, col) else
            deterministic(fallback, -1, False))
        for row in range(n_rows):
            for col in range(n_cols):
                state = mrc(row, col)
                P[state][0] = outcome(row - 1, col, state)
                P[state][1] = outcome(row, col + 1, state)
                P[state][2] = outcome(row + 1, col, state)
                P[state][3] = outcome(row, col - 1, state)
        for action in range(n_actions):
            P[mrc(1, 0)][action] = deterministic(mrc(1, 4), 10, False)
            P[mrc(3, 0)][action] = deterministic(mrc(3, 2), 6, False)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.mrc = mrc
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
    def render_qf(self, qf):
        zf = lambda rc: max(qf(self.mrc(*rc)))
        xs = range(self.n_cols)
        ys = range(self.n_rows)
        xs, ys = np.meshgrid(xs, ys)
        xys = np.dstack((xs, ys))
        zs = np.apply_along_axis(zf, 2, xys)
        ax = sns.heatmap(zs, linewidths=0.2, annot=True)

if __name__ == '__main__':
    import os
    env = GridWorld()
    qf, step = sole_step(env, 0.9)
    # env.render_qf(qf)
    for i in range(301):
        print(i)
        # plt.clf()
        env.render_qf(qf)
        plt.show()
        # plt.draw()
        # plt.pause(0.0000000000001)
        # plt.savefig(os.path.join('pictures', f'{i}.png'))
        step()