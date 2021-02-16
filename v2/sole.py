from v2.utils import argmax
from collections import deque
import numpy as np

def sole_step(env, discount, stochastic=True):
    # Define State Action Space
    n_states = env.n_states + 1 # Plus 1 for Absorbing State
    n_actions = env.n_actions
    n_state_actions = n_states * n_actions
    # Map State-Action
    msa = lambda state, action: state * n_actions + action
    # Build Reward & Transition Matrix
    R = np.zeros(n_state_actions)
    P = np.zeros((n_state_actions, n_states))
    for state in range(n_states - 1):
        for action in range(n_actions):
            for p, s, r, d in env.P[state][action]:
                R[msa(state, action)] += p * r
                if d: s = n_states - 1 # To Absorbing State
                P[msa(state, action), s] += p
    # Handle Absorbing State
    for action in range(n_actions):
        state = n_states - 1
        P[msa(state, action), state] = 1
    # GPI
    I = np.identity(n_state_actions)
    Q = np.zeros(n_state_actions)
    PI = np.zeros((n_states, n_state_actions))
    # Solve for Q
    def solve_Q():
        nonlocal Q
        Q = np.linalg.solve(I - discount * P @ PI, R)
    # Build Policy Matrix
    def build_PI():
        PI.fill(0)
        for state in range(n_states):
            state_action_0 = msa(state, 0)
            state_action_n = msa(state, n_actions)
            if stochastic:
                sub_Q = Q[state_action_0:state_action_n]
                max_value = np.amax(sub_Q)
                indices = np.argwhere(sub_Q == max_value)
                indices += state_action_0
                PI[state][indices] = 1 / len(indices)
            else:
                greedy_state_action = \
                    state_action_0 + np.argmax(
                        Q[state_action_0:state_action_n])
                PI[state, greedy_state_action] = 1
    # Step
    def step():
        build_PI()
        solve_Q()
    # Q Function
    qf = lambda state, action=None: (
        Q[msa(state, 0):msa(state, n_actions)] if action is None
        else Q[msa(state, action)])
    return qf, step

def sole(env, discount, stochastic=False, max_iterations=99, n_prev_policies=2):
    n_states = env.n_states
    qf, step = sole_step(env, discount, stochastic)
    qf_policy = lambda: [argmax(qf(state)) for state in range(n_states)]
    # Loop
    policies = deque(maxlen=n_prev_policies)
    for _ in range(max_iterations):
        step()
        policy = qf_policy()
        if policy in policies: break
        policies.appendleft(policy)
    return qf