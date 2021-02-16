from v2.utils import randbool, argmax
import numpy as np

def act_e_greedy(rand_action, epsilon, qf):
    return lambda state, ts=None: (
        rand_action()
        if randbool(epsilon if ts is None else epsilon(ts))
        else argmax(qf(state)))

def e_greedy(n_actions, epsilon, qf):
    def policy(state, action=None, ts=None):
        e = epsilon if ts is None else epsilon(ts)
        if action is None:
            ps = np.full(n_actions, e / n_actions)
            ps[argmax(qf(state))] += 1 - e
            return ps
        else:
            return (
                (1 - e + e / n_actions)
                if action == argmax(qf(state))
                else (e / n_actions))
    return policy