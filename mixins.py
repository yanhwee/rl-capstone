from collections import deque

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