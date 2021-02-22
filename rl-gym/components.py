import numpy as np
from collections import deque

class Interact:
    def __init__(self, env, agent, gamma_func):
        self.env = env
        self.agent = agent
        self.gamma_func = gamma_func
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None
        self.info = None
        self.reset()
    def reset(self):
        self.done = True
    def step(self, learn=True):
        self.state = self.next_state
        if self.done:
            self.state = self.env.reset()
            self.agent.start(self.state)
        self.action = self.agent.act(self.state)
        self.next_state, self.reward, self.done, self.info = self.env.step(self.action)
        if learn:
            self.agent.observe(self.action, self.reward, self.next_state, self.gamma_func(self.next_state))
        if self.done and learn:
            self.agent.end()

class RingBuffer:
    def __init__(self, maxlen):
        self.data = [None] * maxlen
        self.i = 0
    def clear(self):
        self.data = [None] * len(self.data)
        self.i = 0
    def append(self, value):
        self.data[self.i % len(self.data)] = value
        self.i += 1
    def full(self):
        return self.i >= len(self.data)
    def __len__(self):
        return min(self.i, len(self.data))
    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= 0: key -= len(self)
            if key >= 0 or key < -len(self): raise IndexError
            return self.data[(self.i + key) % len(self.data)]
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            start -= len(self)
            stop -= len(self)
            return [self.data[(self.i + j) % len(self.data)] for j in range(start, stop, step)]
        raise TypeError
    def __setitem__(self, key, value):
        if isinstance(key, int):
            if key >= 0: key -= len(self)
            if key >= 0 or key < -len(self): raise IndexError
            self.data[(self.i + key) % len(self.data)] = value
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            start -= len(self)
            stop -= len(self)
            for j, x in zip(range(start, stop, step), value):
                self.data[(self.i + j) % len(self.data)] = x
    def __repr__(self):
        return f'RingBuffer({self[:]}, maxlen={len(self.data)})'

class Trajectory:
    def __init__(self, length):
        self.rewards = RingBuffer(length)
        self.states = RingBuffer(length)
        self.gammas = RingBuffer(length)
        self.actions = RingBuffer(length)
    def clear(self):
        self.rewards.clear()
        self.states.clear()
        self.gammas.clear()
        self.actions.clear()
    def update(self, reward=None, state=None, gamma=None, action=None):
        if reward is not None: self.rewards[-1] = reward
        if state is not None: self.states[-1] = state
        if gamma is not None: self.gammas[-1] = gamma
        if action is not None: self.actions[-1] = action
    def push_back(self):
        self.rewards.append(None)
        self.states.append(None)
        self.gammas.append(None)
        self.actions.append(None)
    def full(self):
        return self.rewards.full()
    def __len__(self):
        return len(self.rewards)
    def __getitem__(self, key):
        return (
            self.rewards[key], self.states[key],
            self.gammas[key], self.actions[key])

class Trajectory2:
    def __init__(self, length):
        self.rewards = deque(maxlen=length)
        self.states = deque(maxlen=length)
        self.gammas = deque(maxlen=length)
        self.actions = deque(maxlen=length)
    def clear(self):
        self.rewards.clear()
        self.states.clear()
        self.gammas.clear()
        self.actions.clear()
    def update(self, reward=None, state=None, gamma=None, action=None):
        if reward is not None: self.rewards[-1] = reward
        if state is not None: self.states[-1] = state
        if gamma is not None: self.gammas[-1] = gamma
        if action is not None: self.actions[-1] = action
    def push_back(self):
        self.rewards.append(None)
        self.states.append(None)
        self.gammas.append(None)
        self.actions.append(None)
    def full(self):
        return len(self.rewards) == self.rewards.maxlen
    def __len__(self):
        return len(self.rewards)
    def __getitem__(self, key):
        return (
            list(self.rewards)[key], list(self.states)[key],
            list(self.gammas)[key], list(self.actions)[key])

class Returns:
    @staticmethod
    def expected_sarsa(rewards, states, gammas, actions, model, policy):
        returns = rewards[-1] + gammas[-1] * np.dot(
            policy.probability(states[-1], None), 
            model.predict([states[-1]], [None])[0])
        for reward, state, gamma, action in zip(rewards[:-1], states[:-1], gammas[:-1], actions[:-1]):
            p_values = policy.probability(state, None)
            q_values = model.predict([state], [None])[0]
            q_values[action] = returns
            returns = reward + gamma * np.dot(p_values, q_values)
        return returns

class Preprocessing:
    def transform(self, *args, **kwargs):
        raise NotImplementedError

class RLModel:
    def predict(self, states=None, actions=None):
        raise NotImplementedError
    def fit(self, states=None, actions=None, targets=None):
        raise NotImplementedError

class Model:
    def predict(self, X):
        raise NotImplementedError
    def partial_fit(self, X, y, alpha):
        raise NotImplementedError
    @staticmethod
    def _predict(w, X):
        raise NotImplementedError
    @staticmethod
    def _partial_fit(w, X, y, alpha):
        raise NotImplementedError

class Policy:
    @staticmethod
    def choose(probabilities):
        return np.random.choice(len(probabilities), 1, p=probabilities)[0]
    def probability(self, state=None, action=None):
        raise NotImplementedError

class Agent:
    def start(self, state):
        raise NotImplementedError
    def act(self, state):
        raise NotImplementedError
    def observe(self, action, reward, next_state, gamma):
        raise NotImplementedError
    def end(self):
        raise NotImplementedError

class QLearningNStep_DD(Agent):
    def __init__(self, n_step, n_states, n_actions, alpha, epsilon, alpha_decay, epsilon_decay):
        self.trajectory = Trajectory2(n_step + 1)
        self.model = Tabular_DD(n_states, n_actions, alpha)
        self.behavioural_policy = EGreedy(self.model, epsilon)
        self.target_policy = Greedy(self.model)
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay
    def start(self, state):
        self.trajectory.clear()
        self.trajectory.push_back()
        self.trajectory.update(state=state)
    def act(self, state):
        p_values = self.behavioural_policy.probability(state, None)
        return Policy.choose(p_values)
    def observe(self, action, reward, next_state, gamma):
        self.trajectory.update(action=action)
        self.trajectory.push_back()
        self.trajectory.update(reward=reward, state=next_state, gamma=gamma)
        if self.trajectory.full():
            rewards, states, gammas, actions = self.trajectory[1:]
            returns = Returns.expected_sarsa(
                rewards, states, gammas, actions, self.model, self.target_policy)
            _, state, _, action = self.trajectory[0]
            self.model.partial_fit([state], [action], [returns])
        self.behavioural_policy.epsilon = max(0, self.behavioural_policy.epsilon - self.epsilon_decay)
        self.model.model.alpha = max(0, self.model.model.alpha - self.alpha_decay)
    def end(self):
        for i in range(len(self.trajectory) - 1):
            rewards, states, gammas, actions = self.trajectory[i + 1:]
            returns = Returns.expected_sarsa(
                rewards, states, gammas, actions, self.model, self.target_policy)
            _, state, _, action = self.trajectory[i]
            self.model.partial_fit([state], [action], [returns])

class Greedy(Policy):
    def __init__(self, model):
        self.model = model
    def probability(self, state=None, action=None):
        q_values = self.model.predict([state], [None])[0]
        p_values = np.zeros((len(q_values),))
        p_values[np.argmax(q_values)] = 1
        return p_values[action] if action else p_values

class EGreedy(Policy):
    def __init__(self, model, epsilon):
        self.model = model
        self.epsilon = epsilon
    def probability(self, state=None, action=None):
        q_values = self.model.predict([state], [None])[0]
        p_values = np.full((len(q_values),), self.epsilon / len(q_values))
        p_values[np.argmax(q_values)] += 1 - self.epsilon
        return p_values[action] if action else p_values

class OneDiscreteEncode(Preprocessing):
    def __init__(self, nums):
        self.cumprod = np.cumprod(np.concatenate(([1], nums[:-1])))
    def transform(self, X):
        return np.matmul(X, self.cumprod)

class OneHotEncode(Preprocessing):
    def __init__(self, nums):
        self.encoder = OneDiscreteEncode(nums)
        self.n_output = np.prod(nums)
    def transform(self, X):
        values = self.encoder.transform(X)
        ret = np.zeros((len(values), self.n_output))
        ret[range(len(values)),values] = 1
        return ret

class Linear(Model):
    def __init__(self, n, alpha):
        self.w = np.zeros([n])
        self.alpha = alpha
    def predict(self, X):
        return self._predict(self.w, X)
    def partial_fit(self, X, y):
        return self._partial_fit(self.w, X, y, self.alpha)
    @staticmethod
    def _predict(w, X):
        return np.matmul(X, w)
    @staticmethod
    def _partial_fit(w, X, y, alpha):
        errors = y - Linear._predict(w, X)
        gradient = np.matmul(errors, X)
        w += alpha * gradient
        return errors

class Tabular_DD(RLModel):
    def __init__(self, n_states, n_actions, alpha):
        self.n_states = n_states
        self.n_actions = n_actions
        self.encoder = OneHotEncode((n_states, n_actions))
        self.model = Linear(self.encoder.n_output, alpha)
    def predict(self, states=None, actions=None):
        ret = []
        for state, action in zip(states, actions):
            X = [(state, action)] if action else [(state, action) for action in range(self.n_actions)]
            X = self.encoder.transform(X)
            y = self.model.predict(X)
            ret.append(y[0] if action else y)
        return ret
    def partial_fit(self, states=None, actions=None, targets=None):
        X = list(zip(states, actions))
        X = self.encoder.transform(X)
        y = targets
        self.model.partial_fit(X, y)

# class Tabular(Preprocessing):
#     def __init__(self, lows, highs, nums, steps):
#         self.lows = np.array(lows)
#         self.highs = np.array(highs)
#         self.nums = np.array(nums)
#         self.steps = np.array(steps)
#         self.cumprod = np.cumprod(np.concatenate(([1], nums[:-1])))
#     def __call__(self, X):
#         X = np.array(X)
#         for x in X:
#             y = round((x - self.lows) / self.steps)
#             a = 0

#BinarizedTabular

# class SegregatedTabular(RLModel):
#     def __init__(self, state_bounds, action_bounds, model_type):
#         self.state_indices, self.state_sizes, self.state_discrete = self.process_bounds(state_bounds)
#         self.action_indices, self.action_sizes, self.action_discrete = self.process_bounds(actions_bound)
#         self.weights = np.full(
#             np.concatenate((self.state_sizes, self.action_sizes)), 
#             model_type(len(self.state_indices) - len(self.state_sizes)))
#     @staticmethod
#     def process_bounds(bounds):
#         bounds = np.array(bounds)
#         d_indices = np.where(bounds != None)[0]
#         c_indices = np.where(bounds == None)[0]
#         indices = np.concatenate((d_indices, c_indices))
#         sizes = np.array([
#             high - low + 1 for high, low in bounds[d_indices]])
#         return indices, sizes, len(c_indices) == 0
#     def predict(self, states, actions=None):
#         if actions is None: actions = [None] * len(states)
#         for state, action in zip(states, actions):
#             state = state[self.state_indices]
#             action = action[self.action_indices]
#             index = np.concatenate((
#                 state[:len(self.state_sizes)], action[:len(self.action_sizes)]))
#     def fit(self, states, actions=None, targets=None):
#         raise NotImplementedError

# class Tabular(Model):
#     @staticmethod
#     def predict(w, X):
#         return w[tuple(X.T)]
#     @staticmethod
#     def fit(w, X, y, alpha):
#         td_errors = y - Tabular.predict(w, X)
#         for x, td_error in zip(x, td_errors):
#             w[tuple(x)] = alpha * td_error
#         return td_errors

# class TabularLinear(Model):
#     @staticmethod
#     def predict(w, X):
#         w.shape