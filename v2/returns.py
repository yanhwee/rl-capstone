from v2.utils import array_walrus
import numpy as np

def sample_0(states, actions, qf):
    return lambda ts: qf(states[ts], actions[ts])

def sample():
    return lambda ts, g: g

def expected_0(states, qf, policy):
    return lambda ts: \
        np.dot(policy(states[ts], ts=ts), qf(states[ts]))

def expected(states, actions, qf, policy):
    return lambda ts, g: \
        np.dot(policy(states[ts], ts=ts), array_walrus(
            qf(states[ts]), actions[ts], g))

def amax_0(states, qf):
    return lambda ts: max(qf(states[ts]))
    
def amax(states, actions, qf):
    return lambda ts, g: \
        max(array_walrus(qf(states[ts]), actions[ts], g))

def discounted(discount, t0, ts, rewards, g, g_0):
    t1 = t0 + 1
    return rewards[t0] + discount * (
        g(t1, discounted(discount, t1, ts, rewards, g, g_0))
        if t1 < ts else g_0(ts))

def discounted_terminal(discount, t0, ts, rewards, g):
    t1 = t0 + 1
    return rewards[t0] + discount * (
        g(t1, discounted_terminal(discount, t1, ts, rewards, g))
        if t1 < ts else rewards[ts])

def sarsa(discount, t0, ts, states, actions, rewards, qf):
    return discounted(
        discount, t0, ts, rewards, 
        sample(), sample_0(states, actions, qf))

def sarsa_terminal(discount, t0, ts, rewards):
    return discounted_terminal(
        discount, t0, ts, rewards, sample())

def tree(discount, t0, ts, states, actions, rewards, qf, policy):
    return discounted(
        discount, t0, ts, rewards,
        expected(states, actions, qf, policy),
        expected_0(states, qf, policy))

def tree_terminal(discount, t0, ts, states, actions, rewards, qf, policy):
    return discounted_terminal(
        discount, t0, ts, rewards,
        expected(states, actions, qf, policy))

def expected_sarsa(discount, t0, ts, states, rewards, qf, policy):
    return discounted(
        discount, t0, ts, rewards, 
        sample(), expected_0(states, qf, policy))

def expected_sarsa_terminal(discount, t0, ts, rewards):
    return discounted_terminal(
        discount, t0, ts, rewards, sample())

def qlearning(discount, t0, ts, states, actions, rewards, qf):
    return discounted(
        discount, t0, ts, rewards,
        amax(states, actions, qf),
        amax_0(states, qf))

def qlearning_terminal(discount, t0, ts, states, actions, rewards, qf):
    return discounted_terminal(
        discount, t0, ts, rewards,
        amax(states, actions, qf))