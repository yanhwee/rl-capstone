import numpy as np
from collections import deque

class AgentMixin:
    def __init__(self, discount_factor, target_policy, behaviour_policy):
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        self.behaviour_policy = behaviour_policy if behaviour_policy else target_policy
    def act(self, state):
        q_values = self.q_values(state)
        return self.behaviour_policy.choose(q_values)
    def q_values(self, state):
        raise NotImplementedError

class MemoryMixin:
    def __init__(self, n_step):
        self.n_step = n_step
        self.max_length = n_step + 1
        self.rewards = deque(maxlen=self.max_length)
        self.states = deque(maxlen=self.max_length)
        self.actions = deque(maxlen=self.max_length)
    def start(self, state):
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.append(0)
        self.states.append(state)
    def observe(self, action, next_state, reward):
        self.actions.append(action)
        self.learn()
        self.states.append(next_state)
        self.rewards.append(reward)
    def ready(self):
        return len(self.rewards) >= self.max_length
    def empty(self):
        return len(self.rewards) <= 0
    def learn(self):
        raise NotImplementedError

class LinearModelMixin:
    def __init__(self, n_actions, n_features, learning_rate):
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_actions, n_features))
    def q_value(self, state, action):
        return np.dot(self.weights[action], state)
    def q_values(self, state):
        return np.dot(self.weights, state)
    def sgd_update(self, state, action, td_error):
        self.weights[action] += self.learning_rate * td_error * state

class EligibilityTraceMixin:
    def __init__(self, n_actions, n_features, trace_update, trace_decay):
        self.traces = np.zeros((n_actions, n_features))
        self.trace_update = trace_update
        self.trace_decay = trace_decay