import numpy as np

class Model:
    def __init__(self):
        self.weights = None
    def q_value(self, state, action):
        raise NotImplementedError
    def q_values(self, state):
        raise NotImplementedError
    def sgd_update(self, state):
        raise NotImplementedError

