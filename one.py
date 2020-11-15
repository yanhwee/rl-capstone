import numpy as np
from collections import deque
    
class ClassicAgent:
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, behavioural_policy, expectation, 
        trace_update, trace_decay, online, n_step, learning_rate):
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        if behavioural_policy is None:
            self.behavioural_policy = target_policy
        self.expectation = expectation
        self.trace_update = trace_update
        self.trace_decay = trace_decay
        self.online = online
        self.n_step = n_step
        self.learning_rate = learning_rate
        if online:
            self.traces = np.zeros((n_actions, n_features))
        else:
            self.max_length = n_step + 1
            self.td_errors = deque(maxlen=self.max_length)
            self.states = deque(maxlen=self.max_length)
            self.actions = deque(maxlen=self.max_length)
        self.weights = np.zeros((n_actions, n_features))
    def start(self, state):
        if self.online:
            self.rewards.clear()
            self.states.clear()
            self.actions.clear()
            self.rewards.append(0)
            self.states.append(state)
    def act(self, state):
        q_values = self.weights * state
        return self.behavioural_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        self.actions.append(action)
        td_error = 
        if self.expectation:
            
        if self.online:
            pass
        elif len(self.rewards) >= self.max_length:
            pass
        self.states.append(next_state)
        self.rewards.append(reward)
    def end(self):
        pass