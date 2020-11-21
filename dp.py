import gym
import numpy as np
from IPython.display import clear_output
import time, sys
import matplotlib.pyplot as plt
from utils import simple_line, simple_bar, mae, argmax_equal_percent

class DP:
    def __init__(self, env, gamma, limit=999, policy_stop=True, q_value_stop=True, eval_q_table=None):
        # Define State Action Space
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        # Value Iteration
        q_table = np.zeros([n_states, n_actions])
        # Collect Stats
        PI_changes, Q_changes, Q_dt = [], [], []
        q_mae, pi_opt = [], []
        for i in range(limit):
            old_q_table = q_table.copy()
            one = time.perf_counter()
            for state in range(n_states):
                for action, scenarios in env.P[state].items():
                    q_value = 0
                    for p, s, r, d in scenarios:
                        q_value += p * (r + (
                            0 if d else gamma * np.max(q_table[s])))
                    q_table[state, action] = q_value
            two = time.perf_counter()
            # Collect Stats
            new_q_table = q_table
            old_policy = np.argmax(old_q_table, axis=1)
            new_policy = np.argmax(new_q_table, axis=1)
            PI_changes.append((old_policy != new_policy).sum())
            Q_changes.append((old_q_table != new_q_table).sum())
            Q_dt.append(two - one)
            if eval_q_table is not None:
                q_mae.append(mae(q_table, eval_q_table))
                pi_opt.append(
                    argmax_equal_percent(q_table, eval_q_table))
            print(i, end=' ')
            if policy_stop and np.array_equal(new_policy, old_policy): break
            if q_value_stop and np.array_equal(new_q_table, old_q_table): break
        # Save
        self.env, self.q_table, self.PI_changes, self.Q_changes, self.Q_dt, self.q_mae, self.pi_opt = env, q_table, PI_changes, Q_changes, Q_dt, q_mae, pi_opt
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
    def stats(self, title=None):
        PI_changes, Q_changes, Q_dt, q_mae, pi_opt = self.PI_changes, self.Q_changes, self.Q_dt, self.q_mae, self.pi_opt
        simple_bar(Q_changes, 1, 'Changes in Q Values', 'Sweeps', title)
        simple_bar(PI_changes, 1, 'Changes in Policy', 'Sweeps', title)
        simple_bar(Q_dt, 1, 'Time Taken for GPI', 'Sweeps', title)
        if q_mae: simple_bar(q_mae, 1, 'Average State Action Loss', 'Sweeps', title)
        if pi_opt: simple_bar(pi_opt, 1, '% Optimal Action', 'Sweeps', title)
        print('Total Changes in Q Values', np.sum(Q_changes))
        print('Total Changes in Policy', np.sum(PI_changes))
        print('Total Time Taken for GPI', np.sum(Q_dt))
        if q_mae: print('Final Q Values MAE', q_mae[-1])
        if pi_opt: print('Final % Optimal Policy', pi_opt[-1])
    def get_q_table(self):
        return self.q_table.copy()
