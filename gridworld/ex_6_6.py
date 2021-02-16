from v2.env import DiscreteEnv, PreprocessLinearEnv
from v2.features import tab2lin
from v2.agents import sarsa, qlearning, expectedsarsa
from v2.utils import linear_decay_clip
import v2.plot as Plot
from v2.stats import variable_length_mean
from random import randrange, choices
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool
from pprint import pprint

class GridWorld(DiscreteEnv):
    def __init__(self):
        n_rows = 3
        n_cols = 12
        n_states = n_rows * n_cols + 2
        n_actions = 4
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
        start_state = n_states - 2
        end_state = n_states - 1
        for row in range(n_rows):
            for col in range(n_cols):
                state = mrc(row, col)
                P[state][0] = outcome(row - 1, col, state)
                P[state][1] = outcome(row, col + 1, state)
                P[state][2] = outcome(row + 1, col, state)
                P[state][3] = outcome(row, col - 1, state)
        for col in range(1, n_cols - 1):
            state = mrc(n_rows - 1, col)
            P[state][2] = deterministic(start_state, -100, False)
        P[mrc(n_rows - 1, 0)][3] = deterministic(start_state, -1, False)
        P[mrc(n_rows - 1, n_cols - 1)][3] = deterministic(end_state, -1, True)
        P[start_state][0] = deterministic(mrc(n_rows - 1, 0), -1, False)
        P[start_state][1] = deterministic(start_state, -1, False)
        P[start_state][2] = deterministic(start_state, -1, False)
        P[start_state][3] = deterministic(start_state, -1, False)
        P[end_state][0] = deterministic(end_state, 0, True)
        P[end_state][1] = deterministic(end_state, 0, True)
        P[end_state][2] = deterministic(end_state, 0, True)
        P[end_state][3] = deterministic(end_state, 0, True)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.mrc = mrc
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.state = 0
    def rand_state(self):
        return randrange(self.n_states)
    def rand_action(self):
        return randrange(self.n_actions)
    def reset(self):
        self.state = 0
        return self.state
    def step(self, action):
        scenarios = self.P[self.state][action]
        weights = [p for p, s, r, d in scenarios]
        p, s, r, d = choices(scenarios, weights)[0]
        self.state = s
        return s, r, d

if __name__ == '__main__':
    env = GridWorld()
    env = PreprocessLinearEnv(env, tab2lin(env.n_states))
    with ProcessingPool() as p:
        algo_tasks = lambda algo: \
            [p.apipe(
                algo, env, discount=1, train_ts=20000,
                epsilon=linear_decay_clip(0.1, 0.1, 1),
                lr=0.1, nstep=1) for _ in range(100)]
        tasks_get = lambda tasks: \
            [task.get() for task in tasks]
        algo_tasks_get = lambda tasks: \
            zip(*tasks_get(tasks))
        hists_eps_rewards = lambda hists: \
            [hist['eps_rewards'] for hist in hists]
        sarsa_tasks = algo_tasks(sarsa)
        qlearning_tasks = algo_tasks(qlearning)
        exsarsa_tasks = algo_tasks(expectedsarsa)
        s_hists, s_qfs = algo_tasks_get(sarsa_tasks)
        q_hists, q_qfs = algo_tasks_get(qlearning_tasks)
        x_hists, x_qfs = algo_tasks_get(exsarsa_tasks)
        s_eps_rewards = hists_eps_rewards(s_hists)
        q_eps_rewards = hists_eps_rewards(q_hists)
        x_eps_rewards = hists_eps_rewards(x_hists)
        s_mean_rewards = variable_length_mean(*s_eps_rewards)
        q_mean_rewards = variable_length_mean(*q_eps_rewards)
        x_mean_rewards = variable_length_mean(*x_eps_rewards)
        plt.plot(s_mean_rewards, label='sarsa')
        plt.plot(q_mean_rewards, label='qlearning')
        plt.plot(x_mean_rewards, label='expectedsarsa')
        plt.ylim(-75, -10)
        plt.title('Cliff Walking')
        plt.ylabel('Sum of Rewards during Episode')
        plt.xlabel('Episodes')
        plt.show()

    # Plot.eps_rewards(history['eps_rewards'])
    # plt.plot(history['eps_rewards'])
    # plt.plot(history2['eps_rewards'])
    # plt.ylim(-100, 0)
    # plt.show()
    # Plot.from_histories([history, history2])