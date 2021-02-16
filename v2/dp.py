from v2.utils import argmax
from itertools import cycle
from collections import deque
import numpy as np

def dp_step(env, discount):
    # Define State Action Space
    n_states = env.n_states
    n_actions = env.n_actions
    # Value Iteration
    q_table = np.zeros((n_states, n_actions))
    # Step
    def step():
        for state in range(n_states):
            for action in range(n_actions):
                q_value = 0
                for p, s, r, d in env.P[state][action]:
                    q_value += p * (r + (
                        0 if d else discount * max(q_table[s])))
                q_table[state, action] = q_value
    # Q Function
    qf = lambda state, action=None: (
        q_table[state] if action is None
        else q_table[state,action])
    return qf, step

def dp_micro_step(env, discount):
    # Define State Action Space
    n_states = env.n_states
    n_actions = env.n_actions
    # Value Iteration
    q_table = np.zeros((n_states, n_actions))
    # Step
    state_cycle = cycle(range(n_states))
    def micro_step():
        state = next(state_cycle)
        for action in range(n_actions):
            q_value = 0
            for p, s, r, d in env.P[state][action]:
                q_value += p * (r + (
                    0 if d else discount * max(q_table[s])))
            q_table[state, action] = q_value
    # Q Function
    qf = lambda state, action=None: (
        q_table[state,action] if action
        else q_table[state])
    return qf, micro_step

def dp(env, discount, max_iterations=99, n_prev_policies=2):
    n_states = env.n_states
    qf, step = dp_step(env, discount)
    qf_policy = lambda: [argmax(qf(state)) for state in range(n_states)]
    # Loop
    policies = deque(maxlen=n_prev_policies)
    for _ in range(max_iterations):
        step()
        policy = qf_policy()
        if policy in policies: break
        policies.appendleft(policy)
    return qf