from collections import deque

class Returns:
    def __init__(self):
        pass
    def __call__(self, rewards, states, actions):
        pass

class Sarsa(Returns):
    def __init__(self, gamma, model):
        self.gamma = gamma
        self.model = model
    def __call__(self, rewards, states, actions):
        pass

class ExpectedSarsa(Returns):
    def __init__(self, gamma, model, policy):
        self.gamma = gamma
        self.model = model
        self.policy = policy
    def __call__(self, rewards, states, actions):
        pass