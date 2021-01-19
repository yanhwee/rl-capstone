from v1.utils import comb, prod, key_map
from itertools import combinations_with_replacement, product, accumulate
import operator as op
import numpy as np

class Preprocessor:
    def __init__(self):
        self.n_output = None
    def __call__(self, features):
        raise NotImplementedError

class Normalise(Preprocessor):
    def __init__(self, n_features, lows, highs):
        self.lows = lows
        self.factor = (highs - lows) ** -1
        self.n_output = n_features
    def __call__(self, features):
        return ((features - self.lows) * self.factor).clip(0, np.nextafter(1, 0))

class Polynomial(Preprocessor):
    def __init__(self, n_features, degree):
        self.n_output = comb(n_features + degree, degree) # No minus one because include bias
        self.degree = degree
    def __call__(self, features):
        output = np.zeros(self.n_output)
        for i, combo in enumerate(
                combinations_with_replacement(
                    np.concatenate(([1], features)), self.degree)):
            output[i] = prod(combo)
        return output

class Fourier(Preprocessor):
    def __init__(self, n_features, degree):
        self.c = np.array(list(product(range(degree), repeat=n_features))) * np.pi
        self.n_output = degree ** n_features
    def __call__(self, features):
        return np.cos(np.dot(self.c, features))

class FullAggregation(Preprocessor):
    def __init__(self, n_features, intervals):
        self.width = 1 / intervals
        self.key = key_map([intervals] * n_features)
        self.identity = np.eye(intervals ** n_features)
        self.n_output = intervals ** n_features
    def __call__(self, features):
        index = round(np.dot(self.key, features // self.width))
        return self.identity[index]

class PartialAggregation(Preprocessor):
    def __init__(self, n_features, intervals):
        self.width = 1 / intervals
        self.identity = np.eye(intervals)
        self.n_output = intervals * n_features
    def __call__(self, features):
        indices = (features // self.width).astype(int)
        return self.identity[indices].flatten()