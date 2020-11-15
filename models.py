import numpy as np

class Model:
    def q_value(self, state, action):
        raise NotImplementedError
    def q_values(self, state):
        raise NotImplementedError
    def sgd_update(self, state):
        raise NotImplementedError

class LinearModel(Model):
    def __init__(self, n_actions, n_features, learning_rate):
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_actions, n_features))
    def q_value(self, state, action):
        return np.dot(self.weights[action], state)
    def q_values(self, state):
        return np.dot(self.weights, state)
    def sgd_update(self, td_error):
        self.weights += self.learning_rate * td_error * self.weights