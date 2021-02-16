import numpy as np

def weights(n_features, n_actions):
    return np.zeros((n_actions, n_features))

def qf(weights):
    return lambda state, action=None: (
        np.dot(weights, state) if action is None else
        np.dot(weights[action], state))

def update_q_td_error(weights, lr, state, action, td_error):
    weights[action] += lr * td_error * state

def update_q_target(weights, lr, state, action, target):
    update_q_td_error(
        weights, lr, state, action,
        target - qf(weights)(state, action))

# def q_values(weights):
#     return lambda state: np.dot(weights, state)

# def q_value(weights):
#     return lambda state, action: np.dot(weights[action], state)

# def q_function(weights):
#     return lambda state, action=None: (
#         q_value(weights)(state, action)
#         if action else q_values(weights)(state))