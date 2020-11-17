import numpy as np

class Policy:
    def choose(self, values):
        pass
    def weighted_sum(self, a, b):
        pass
    def probability(self, action, q_values):
        pass
    # def probabilities(self, values):
    #     pass
    # def weighted_sum(self, values):
    #     pass

class Greedy(Policy):
    def __init__(self):
        pass
    def choose(self, values):
        return np.argmax(values)
    def weighted_sum(self, a, b):
        return b[np.argmax(a)]
    def probability(self, action, q_values):
        return int(action == np.argmax(q_values))

class EGreedy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def choose(self, values):
        return (
            np.random.randint(len(values))
            if np.random.random() < self.epsilon
            else np.argmax(values))
    def weighted_sum(self, a, b):
        return (
            np.sum(b) * (self.epsilon / len(a)) +
            b[np.argmax(a)] * (1 - self.epsilon))
    def probability(self, action, q_values):
        return (
            (1 - self.epsilon) * int(action == np.argmax(q_values)) + 
            (self.epsilon / len(q_values)))

# class Greedy(Policy):
    # def weighted_sum(self, values):
    #     return np.amax(values)

# class EGreedy(Policy):
    # def probabilities(self, values):
    #     ps = np.full(len(values), self.epsilon / len(values))
    #     ps[np.argmax(values)] += 1 - self.epsilon
    #     return ps
    # def weighted_sum(self, values):
    #     return (
    #         np.sum(values) * self.epsilon + 
    #         np.amax(values) * (1 - self.epsilon))