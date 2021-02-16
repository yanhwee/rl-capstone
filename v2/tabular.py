import numpy as np

def weights(n_states, n_actions):
    return np.zeros((n_states, n_actions))

def qf(weights):
    def qf2(state=None, action=None):
        if state is None:
            return weights
        elif action is None:
            return weights[state]
        else:
            return weights[state, action]
    return qf2

def update_q_td_error(weights, lr, state, action, td_error):
    weights[state, action] += lr * td_error

def update_q_target(weights, lr, state, action, target):
    update_q_td_error(
        weights, lr, state, action,
        target - qf(weights)(state, action))