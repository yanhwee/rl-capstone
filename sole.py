import gym
from gym import spaces
import numpy as np
from IPython.display import clear_output
import time, sys
import matplotlib.pyplot as plt
from utils import simple_line, simple_bar, mae, argmax_equal_percent

class SOLE:
    def __init__(self, env, gamma, limit=20, eval_q_table=None):
        # Define State Action Space
        n_states = env.observation_space.n + 1
        n_actions = env.action_space.n
        n_state_actions = n_states * n_actions
        # Map State-Action
        def msa(state, action):
            return state * n_actions + action
        # Build Reward & Transition Matrix
        R = np.zeros([n_state_actions])
        P = np.zeros([n_state_actions, n_states])
        for state in range(n_states - 1):
            for action in range(n_actions):
                for p, s, r, d in env.P[state][action]:
                    R[msa(state, action)] += p * r
                    if d: s = n_states - 1 # To Absorbing State
                    P[msa(state, action), s] = p
        # Handle Absorbing State
        for action in range(n_actions):
            state = n_states - 1
            P[msa(state, action), state] = 1
        # GPI
        Q = np.zeros([n_state_actions])
        PI = np.zeros([n_states, n_state_actions])
        I = np.identity(n_state_actions)
        for state in range(n_states):
            PI[state, msa(state, 0)] = 1
        # Collect Stats
        PI_changes, Q_changes = [], []
        PI_dt, Q_dt = [], []
        q_mae, pi_opt = [], []
        for i in range(limit):
            # Re-evaluate Q Values
            old_Q = Q.copy()
            one = time.perf_counter()
            Q = np.linalg.solve(I - gamma * P @ PI, R)
            two = time.perf_counter()
            # Re-build Policy Matrix
            old_PI = PI.copy()
            three = time.perf_counter()
            PI.fill(0)
            for state in range(n_states):
                state_action_0 = msa(state, 0)
                state_action_n = msa(state, n_actions)
                greedy_state_action = \
                    state_action_0 + np.argmax(
                        Q[state_action_0:state_action_n])
                PI[state, greedy_state_action] = 1
            four = time.perf_counter()
            # Collect Stats
            PI_changes.append((old_PI != PI).sum() / 2)
            Q_changes.append((old_Q != Q).sum())
            Q_dt.append(two - one)
            PI_dt.append(four - three)
            if eval_q_table is not None:
                q_table = self._get_q_table(n_states, n_actions, msa, Q)
                q_mae.append(mae(q_table, eval_q_table))
                pi_opt.append(
                    argmax_equal_percent(q_table, eval_q_table))
            print(i, end=' ')
            if np.array_equal(PI, old_PI): break
        # Save
        self.env, self.gamma, self.n_states, self.n_actions, self.n_state_actions, self.msa, self.R, self.P, self.Q, self.PI, self.PI_changes, self.Q_changes, self.PI_dt, self.Q_dt, self.q_mae, self.pi_opt = env, gamma, n_states, n_actions, n_state_actions, msa, R, P, Q, PI, PI_changes, Q_changes, PI_dt, Q_dt, q_mae, pi_opt
    def test(self, delay, limit=None):
        # Get
        env, n_actions, msa, Q, PI = self.env, self.n_actions, self.msa, self.Q, self.PI
        def render(i):
            clear_output(wait=True)
            env.render()
            print(i)
            time.sleep(delay)
        # Loop
        state = env.reset()
        render(0)
        for i in range(limit if limit else sys.maxsize):
            state_action_0 = msa(state, 0)
            state_action_n = msa(state, n_actions)
            action = np.argmax(PI[state, state_action_0:state_action_n])
            action = np.argmax(Q[state_action_0:state_action_n])
            state, reward, done, info = env.step(action)
            render(i)
            if done: break
        env.close()
    def stats(self, title=None):
        q_mae, pi_opt, Q_changes, PI_changes, PI_dt, Q_dt = self.q_mae, self.pi_opt, self.Q_changes, self.PI_changes, self.PI_dt, self.Q_dt
        simple_bar(Q_changes, 1, 'Changes in Q Values', 'Iterations', title)
        simple_bar(PI_changes, 1, 'Changes in Policy', 'Iterations', title)
        simple_bar(np.add(PI_dt, Q_dt), 1, 'Time Taken for GPI', 'Iterations', title)
        if q_mae: simple_bar(q_mae, 1, 'Average State Action Loss', 'Sweeps', title)
        if pi_opt: simple_bar(pi_opt, 1, '% Optimal Action', 'Sweeps', title)
        print('Total Changes in Q Values', np.sum(Q_changes))
        print('Total Changes in Policy', np.sum(PI_changes))
        print('Total Time Taken for GPI', np.sum(np.add(PI_dt, Q_dt)))
        if q_mae: print('Final Q Values MAE', q_mae[-1])
        if pi_opt: print('Final % Optimal Policy', pi_opt[-1])
    def get_q_table(self):
        n_states, n_actions, msa, Q = self.n_states, self.n_actions, self.msa, self.Q
        return self._get_q_table(n_states, n_actions, msa, Q)
    @staticmethod
    def _get_q_table(n_states, n_actions, msa, Q):
        n_states -= 1
        q_table = np.zeros([n_states, n_actions])
        for state in range(n_states):
            for action in range(n_actions):
                q_table[state, action] = Q[msa(state, action)]
        return q_table

    # def slow_policy_train(self, limit=99):
    #     env, gamma, n_states, n_actions, n_state_actions, msa, R, P, Q, PI, PI_changes, Q_changes, PI_dt, Q_dt = self.env, self.gamma, self.n_states, self.n_actions, self.n_state_actions, self.msa, self.R, self.P, self.Q, self.PI, self.PI_changes, self.Q_changes, self.PI_dt, self.Q_dt
    #     I = np.identity(n_state_actions)
    #     for i in range(limit):
    #         # Re-evaluate Q Values
    #         old_Q = Q.copy()
    #         one = time.perf_counter()
    #         Q = np.linalg.solve(I - gamma * P @ PI, R)
    #         two = time.perf_counter()
    #         # Re-build Policy Matrix
    #         old_PI = PI.copy()
    #         three = time.perf_counter()
    #         # PI.fill(0)
    #         for state in range(n_states):
    #             state_action_0 = msa(state, 0)
    #             state_action_n = msa(state, n_actions)
    #             greedy_state_action = \
    #                 state_action_0 + np.argmax(
    #                     Q[state_action_0:state_action_n])
    #             if old_PI[state, greedy_state_action] == 0:
    #                 PI[state].fill(0)
    #                 PI[state, greedy_state_action] = 1
    #                 break
    #         four = time.perf_counter()
    #         # Collect Stats
    #         PI_changes.append((old_PI != PI).sum() / 2)
    #         Q_changes.append((old_Q != Q).sum())
    #         Q_dt.append(two - one)
    #         PI_dt.append(four - three)
    #         print(i, end=' ')
    #         if np.array_equal(PI, old_PI): break

    # def stats_PI_changes(self, title):
    #     simple_line(self.PI_changes, 1, 'Changes in Policy', 'Iterations', title)
    # def stats_q_changes(self, title):
    #     simple_line(self.Q_changes, 1, 'Changes in Q Values', 'Iterations', title)
    # def stats_PI_durations(self, title):
    #     simple_line(self.PI_dt, 1, 'Time Taken for Policy Improvement', 'Iterations', title)
    # def stats_Q_durations(self, title):
    #     simple_line(self.Q_dt, 1, 'Time Taken for Policy Evaluation', 'Iterations', title)
    # def stats_PI_timings(self, title):
    #     simple_line(np.cumsum(self.PI_dt), 1, 'Accumulated Time Taken for Policy Improvement', 'Iterations', title)
    # def stats_Q_timings(self, title):
    #     simple_line(np.cumsum(self.Q_dt), 1, 'Accumulated Time Taken for Policy Evaluation', 'Iterations', title)
    # def stats_PI_Q_durations(self, title):
    #     simple_line(np.add(self.PI_dt, self.Q_dt), 1, 'Time Taken for GPI', 'Iterations', title)
    # def stats_PI_Q_timings(self, title):
    #     simple_line(np.cumsum(np.add(self.PI_dt, self.Q_dt)), 1, 'Accumulated Time Taken for GPI', 'Iterations', title)
