import numpy as np
from mixins import MemoryMixin
from models import LinearModel

class Agent:
    def start(self, state):
        raise NotImplementedError
    def act(self, state):
        raise NotImplementedError
    def observe(self, action, next_state, reward):
        raise NotImplementedError
    def end(self):
        raise NotImplementedError

class Sarsa(Agent, MemoryMixin):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, n_step, learning_rate):
        MemoryMixin.__init__(self, n_step)
        self.model = LinearModel(n_actions, n_features, learning_rate)
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        self.max_discount_factor = self.discount_factor ** n_step
        self.returns = 0
        self.bootstrap = 0
    def start(self, state):
        MemoryMixin.start(self, state)
    def act(self, state):
        q_values = self.model.q_values(state)
        return self.target_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        MemoryMixin.observe(self, action, next_state, reward)
    def learn(self):
        if self.ready():
            self.returns -= self.rewards[0]
            self.returns -= self.bootstrap
            self.returns += self.rewards[-1] * self.max_discount_factor
            self.returns /= self.discount_factor
            self.bootstrap = self.model.q_value(self.states[-1], self.actions[-1])
            self.bootstrap *= self.max_discount_factor
            self.returns += self.bootstrap
            value = self.model.q_value(self.states[0], self.actions[0])
            td_error = self.returns - value
            self.sgd_update(td_error)
        else:
            self.returns += self.rewards[-1] * self.max_discount_factor
            self.returns /= self.discount_factor
    def end(self):
        self.actions.append(None)
        self.returns -= self.bootstrap
        self.returns += self.rewards[-1] * self.max_discount_factor
        for _ in range(self.n_step):
            reward = self.rewards.popleft()
            state = self.states.popleft()
            action = self.actions.popleft()
            self.returns -= reward
            self.returns /= self.discount_factor
            value = self.q_value(state, action)
            td_error = self.returns - value
            self.sgd_update(td_error)

class ExpectedSarsa(Agent, MemoryMixin, LinearModelMixin):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, behaviour_policy, n_step, learning_rate):
        MemoryMixin.__init__(self, n_step)
        LinearModelMixin.__init__(self, n_actions, n_features, learning_rate)
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        self.behaviour_policy = behaviour_policy if behaviour_policy else target_policy
    def start(self, state):
        MemoryMixin.start(self, state)
    def act(self, state):
        q_values = self.q_values(state)
        return self.behaviour_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        MemoryMixin.observe(self, action, next_state, reward)
    def returns(self):
        returns = self.q_value(self.states[-1], self.actions[-1])
        for reward, state, action in zip(self.rewards[:0:-1], self.states[:0:-1], self.actions[:0:-1]):
            q_values = self.q_values(state)
            q_values[action] = returns
            returns = reward + self.discount_factor * self.target_policy.weighted_sum(q_values)
        return returns
    def learn(self):
        if self.ready():
            value = self.q_value(self.states[0], self.actions[0])
            td_error = self.returns() - value
            self.sgd_update(td_error)
    def end(self):
        self.actions.append(None)
        for _ in range(self.n_step):
            value = self.q_value(self.states[0], self.actions[0])
            td_error = self.returns() - value
            self.sgd_update(td_error)
            self.rewards.popleft()
            self.states.popleft()
            self.actions.popleft()

class EligibilityTraceMixin:
    def __init__(self, n_actions, n_features):
        self.traces = np.zeros((n_actions, n_features))

class OnlineSarsa(Agent, EligibilityTraceMixin, LinearModelMixin):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, trace_update, trace_decay, learning_rate):
        EligibilityTraceMixin.__init__(self, n_actions, n_features)
        LinearModelMixin.__init__(self, n_actions, n_features, learning_rate)
        self.discount_factor = discount_factor
        self.target_policy = target_policy

# class Sarsa(AgentNStep):
#     def __init__(
#         self, n_actions, n_features, discount_factor, 
#         target_policy, n_step, learning_rate):
#         super().__init__(n_step)
#         self.discount_factor = discount_factor
#         self.target_policy = target_policy
#         self.learning_rate = learning_rate
#         self.weights = np.zeros((n_actions, n_features))
#         self.max_discount_factor = self.discount_factor ** n_step
#         self.returns = 0
#         self.bootstrap = 0
#     def act(self, state):
#         q_values = self.weights * state
#         return self.target_policy.choose(q_values)
#     def learn(self):
#         if self.ready():
#             self.bootstrap = self.weights[self.actions[-1]] * self.state
#             self.bootstrap *= self.max_discount_factor
#             self.returns -= self.rewards[0]
#             self.returns -= self.bootstrap
#             self.returns += self.rewards[-1] * self.max_discount_factor
#             self.returns /= self.discount_factor
#             self.returns += self.bootstrap
#         else:
#             self.returns += self.rewards[-1] * self.max_discount_factor
#             self.returns /= self.discount_factor

# class Sarsa(Agent):
#     def __init__(
#         self, n_actions, n_features, discount_factor, 
#         target_policy, n_step, learning_rate)
#         self.n_actions = n_actions
#         self.n_features = n_features
#         self.discount_factor = discount_factor
#         self.target_policy = target_policy
#         self.n_step = n_step
#         self.learning_rate = learning_rate
#         self.max_length = n_step + 1
#         self.rewards = deque(maxlen=self.max_length)
#         self.states = deque(maxlen=self.max_length)
#         self.actions = deque(maxlen=self.max_length)
#         self.weights = np.zeros((n_actions, n_features))
#     def start(self, state):
#         self.rewards.clear()
#         self.states.clear()
#         self.actions.clear()
#         self.rewards.append(0)
#         self.states.append(state)
#     def act(self, state):
#         q_values = self.weights * state
#         return self.target_policy.choose(q_values)
#     def observe(self, action, next_state, reward):
#         self.actions.append(action)
#     def end(self):
#         pass

# class Sarsa(Agent):
#     def __init__(self, n_step, gamma, policy, model):
#         self.n_step = n_step
#         self.gamma = gamma
#         self.max_gamma = gamma ** (n_step + 1)
#         self.policy = policy
#         self.model = model
#         self.rewards = deque(maxlen=n_step + 1)
#         self.states = deque(maxlen=n_step + 1)
#         self.actions = deque(maxlen=n_step + 1)
#         self.returns = 0
#     def start(self, state):
#         self.rewards.append(0)
#         self.states.append(state)
#     def act(self, state):
#         pass
#     def observe(self, action, next_state, reward):
#         self.actions.append(action)
#         self.returns += self.max_gamma * self.model.predict((*self.states[-1], *self.actions[-1]))
#         if len(self.rewards) == self.rewards.maxlen:
#             self.returns -= self.rewards[0]
#             self.returns /= self.gamma
#             self.model.learn()
#         self.rewards.append(reward)
#         self.states.append(next_state)
#     def end(self):
#         pass