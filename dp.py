import gym
import numpy as np
from IPython.display import clear_output
import time, sys
import matplotlib.pyplot as plt
from utils import simple_line, simple_bar

class DP:
    def __init__(self, env, gamma, limit=9999, policy_stop=True, q_value_stop=True):
        # Define State Action Space
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        # Value Iteration
        q_table = np.zeros([n_states, n_actions])
        # Collect Stats
        PI_changes, Q_changes, Q_dt = [], [], []
        for i in range(limit):
            old_q_table = q_table.copy()
            one = time.perf_counter()
            for state in range(n_states):
                for action, scenarios in env.P[state].items():
                    q_value = 0
                    for p, s, r, _ in scenarios:
                        q_value += p * (r + gamma * np.max(q_table[s]))
                    q_table[state, action] = q_value
            two = time.perf_counter()
            # Collect Stats
            new_q_table = q_table
            old_policy = np.argmax(old_q_table, axis=1)
            new_policy = np.argmax(new_q_table, axis=1)
            PI_changes.append((old_policy != new_policy).sum())
            Q_changes.append((old_q_table != new_q_table).sum())
            Q_dt.append(two - one)
            print(i, end=' ')
            if policy_stop and np.array_equal(new_policy, old_policy): break
            if q_value_stop and np.array_equal(new_q_table, old_q_table): break
        # Save
        self.env, self.q_table, self.PI_changes, self.Q_changes, self.Q_dt = env, q_table, PI_changes, Q_changes, Q_dt
    def test(self, delay, limit=None):
        env, q_table = self.env, self.q_table
        def render(i):
            clear_output(wait=True)
            env.render()
            print(i)
            time.sleep(delay)
        state = env.reset()
        render(0)
        for i in range(limit if limit else sys.maxsize):
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
            render(i)
            if done: break
        env.close()
    def stats(self, title):
        PI_changes, Q_changes, Q_dt = self.PI_changes, self.Q_changes, self.Q_dt
        simple_bar(Q_changes, 1, 'Changes in Q Values', 'Sweeps', title)
        simple_bar(PI_changes, 1, 'Changes in Policy', 'Sweeps', title)
        simple_bar(Q_dt, 1, 'Time Taken for GPI', 'Sweeps', title)
        print('Total Changes in Q Values', np.sum(Q_changes))
        print('Total Changes in Policy', np.sum(PI_changes))
        print('Total Time Taken for GPI', np.sum(Q_dt))