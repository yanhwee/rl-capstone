import numpy as np

class Policy:
    def __init__(self):
        pass
    def choose(self, values):
        pass
    def weighted_sum(self, values):
        pass
    # def probabilities(self, values):
    #     pass

class Greedy(Policy):
    def __init__(self):
        pass
    def choose(self, values):
        return np.argmax(values)
    def weighted_sum(self, values):
        return np.amax(values)

class EGreedy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def choose(self, values):
        return (
            np.random.randint(len(values))
            if np.random.random() < self.epsilon
            else np.argmax(values))
    def weighted_sum(self, values):
        return (
            np.sum(values) * self.epsilon + 
            np.amax(values) * (1 - self.epsilon))


# class EGreedy(Policy):
    # def probabilities(self, values):
    #     ps = np.full(len(values), self.epsilon / len(values))
    #     ps[np.argmax(values)] += 1 - self.epsilon
    #     return ps