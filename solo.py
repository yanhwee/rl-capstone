import numpy as np
from collections import deque
from itertools import islice

class Agent:
    def start(self, state):
        raise NotImplementedError
    def act(self, state):
        raise NotImplementedError
    def observe(self, action, next_state, reward):
        raise NotImplementedError
    def end(self):
        raise NotImplementedError

class SarsaAgent(Agent):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, n_step, learning_rate):
        # Agent
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        self._max_discount_factor = discount_factor ** n_step
        self._returns = 0
        self._bootstrap = 0
        # Memory
        self.n_step = n_step
        self.max_length = n_step + 1
        self.rewards = deque(maxlen=self.max_length)
        self.states = deque(maxlen=self.max_length)
        self.actions = deque(maxlen=self.max_length)
        # Model
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_actions, n_features))
    def start(self, state):
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.append(0)
        self.states.append(state)
    def act(self, state):
        q_values = self.q_values(state)
        return self.target_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        self.actions.append(action)
        self.learn()
        self.states.append(next_state)
        self.rewards.append(reward)
    def learn(self):
        if self.ready():
            self.update_weights()
    def end(self):
        self.actions.append(None)
        for _ in range(self.memory_length() - 1):
            self.update_weights()
            self.rewards.popleft()
            self.states.popleft()
            self.actions.popleft()
    # Agent
    def returns(self):
        # return _returns
        returns = (
            0 if self.actions[-1] is None 
            else self.q_value(self.states[-1], self.actions[-1]))
        for reward in list(self.rewards)[:0:-1]:
            returns = reward + self.discount_factor * returns
        return returns
    def td_error(self):
        state, action = self.states[0], self.actions[0]
        return self.returns() - self.q_value(state, action)
    # def returns_replace_last(self):
    #     self._returns -= self._bootstrap
    #     self._returns += self.rewards[-1] * self._max_discount_factor
    # def returns_shift_left(self):
    #     self._returns -= self.rewards[0]
    #     self._returns /= self.discount_factor
    # def update_boostrap(self):
    #     self._bootstrap = self.q_value(self.states[-1], self.actions[-1])
    #     self._bootstrap *= self._max_discount_factor
    # def returns_add_bootstrap(self):
    #     self._returns += self._bootstrap
    def update_weights(self):
        state, action = self.states[0], self.actions[0]
        self.weights[action] += self.learning_rate * self.td_error() * state
    # Memory
    def ready(self):
        return len(self.rewards) >= self.max_length
    def memory_length(self):
        return len(self.rewards)
    # Model
    def q_value(self, state, action):
        return np.dot(self.weights[action], state)
    def q_values(self, state):
        return np.dot(self.weights, state)

class TreeAgent(Agent):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, behaviour_policy, n_step, learning_rate):
        # Agent
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        self.behaviour_policy = behaviour_policy
        # Memory
        self.n_step = n_step
        self.max_length = n_step + 1
        self.rewards = deque(maxlen=self.max_length)
        self.states = deque(maxlen=self.max_length)
        self.actions = deque(maxlen=self.max_length)
        # Model
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_actions, n_features))
    def start(self, state):
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.append(0)
        self.states.append(state)
    def act(self, state):
        q_values = self.q_values(state)
        return self.behaviour_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        self.actions.append(action)
        self.learn()
        self.states.append(next_state)
        self.rewards.append(reward)
    def learn(self):
        if self.ready():
            self.update_weights()
    def end(self):
        self.actions.append(None)
        for _ in range(self.memory_length() - 1):
            self.update_weights()
            self.rewards.popleft()
            self.states.popleft()
            self.actions.popleft()
    # Agent
    def returns(self):
        returns = (
            0 if self.actions[-1] is None
            else self.q_value(self.states[-1], self.actions[-1]))
        for reward, state, action in list(zip(self.rewards, self.states, self.actions))[:0:-1]:
            q_values = self.q_values(state)
            values = q_values.copy()
            values[action] = returns
            returns = reward + self.discount_factor * self.target_policy.weighted_sum(q_values, values)

            probabilities = self.target_policy.probabilities(q_values)
            q_values[action] = returns
            np.dot(probabilities, q_values)

            index = argmax(q_values)
            q_values[action] = returns
            return q_values[index]

            q_values[action] = returns
            returns = reward + self.discount_factor * self.target_policy.weighted_sum(q_values)
        return returns
    def td_error(self):
        state, action = self.states[0], self.actions[0]
        return self.returns() - self.q_value(state, action)
    def update_weights(self):
        state, action = self.states[0], self.actions[0]
        self.weights[action] += self.learning_rate * self.td_error() * state
    # Memory
    def ready(self):
        return len(self.rewards) >= self.max_length
    def memory_length(self):
        return len(self.rewards)
    # Model
    def q_value(self, state, action):
        return np.dot(self.weights[action], state)
    def q_values(self, state):
        return np.dot(self.weights, state)

class SarsaAccAgent(Agent):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, trace_decay, learning_rate):
        # Agent
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        # Memory
        self.max_length = 2
        self.rewards = deque(maxlen=self.max_length)
        self.states = deque(maxlen=self.max_length)
        self.actions = deque(maxlen=self.max_length)
        # Eligibility Trace
        self.trace_decay = trace_decay
        self.traces = np.zeros((n_actions, n_features))
        # Model
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_actions, n_features))
    def start(self, state):
        self.traces.fill(0)
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.append(0)
        self.states.append(state)
    def act(self, state):
        q_values = self.q_values(state)
        return self.target_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        self.actions.append(action)
        self.learn()
        self.rewards.append(reward)
        self.states.append(next_state)
    def learn(self):
        if self.ready():
            self.update_trace()
            self.update_weights()
    def end(self):
        self.actions.append(None)
        if self.memory_length() == 2:
            self.update_trace()
            self.update_weights()
    # Agent
    def returns(self):
        returns = (
            0 if self.actions[-1] is None 
            else self.q_value(self.states[-1], self.actions[-1]))
        for reward in list(self.rewards)[:0:-1]:
            returns = reward + self.discount_factor * returns
        return returns
    def td_error(self):
        state, action = self.states[0], self.actions[0]
        return self.returns() - self.q_value(state, action)
    def update_trace(self):
        self.traces *= self.discount_factor * self.trace_decay
        if self.actions[-1] is not None:
            self.traces[self.actions[-1]] += self.states[-1]
    def update_weights(self):
        self.weights += self.learning_rate * self.td_error() * self.traces
    # Memory
    def ready(self):
        return len(self.rewards) >= self.max_length
    def memory_length(self):
        return len(self.rewards)
    # Model
    def q_value(self, state, action):
        return np.dot(self.weights[action], state)
    def q_values(self, state):
        return np.dot(self.weights, state)

class SarsaDutchAgent(Agent):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, trace_decay, learning_rate):
        # Agent
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        # Memory
        self.max_length = 2
        self.rewards = deque(maxlen=self.max_length)
        self.states = deque(maxlen=self.max_length)
        self.actions = deque(maxlen=self.max_length)
        # Eligibility Trace
        self.trace_decay = trace_decay
        self.traces = np.zeros((n_actions, n_features))
        # Model
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_actions, n_features))
    def start(self, state):
        self.traces.fill(0)
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.append(0)
        self.states.append(state)
    def act(self, state):
        q_values = self.q_values(state)
        return self.target_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        self.actions.append(action)
        self.learn()
        self.rewards.append(reward)
        self.states.append(next_state)
    def learn(self):
        if self.ready():
            self.update_trace()
            self.update_weights()
    def end(self):
        self.actions.append(None)
        if self.memory_length() == 2:
            self.update_trace()
            self.update_weights()
    # Agent
    def returns(self):
        returns = (
            0 if self.actions[-1] is None 
            else self.q_value(self.states[-1], self.actions[-1]))
        for reward in list(self.rewards)[:0:-1]:
            returns = reward + self.discount_factor * returns
        return returns
    def td_error(self):
        state, action = self.states[0], self.actions[0]
        return self.returns() - self.q_value(state, action)
    def update_trace(self):
        self.traces *= self.discount_factor * self.trace_decay
        if self.actions[-1] is not None:
            self.traces[self.actions[-1]] = self.states[-1]
    def update_weights(self):
        self.weights += self.learning_rate * self.td_error() * self.traces
    # Memory
    def ready(self):
        return len(self.rewards) >= self.max_length
    def memory_length(self):
        return len(self.rewards)
    # Model
    def q_value(self, state, action):
        return np.dot(self.weights[action], state)
    def q_values(self, state):
        return np.dot(self.weights, state)

class TreeAccAgent(Agent):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, behaviour_policy, trace_decay, learning_rate):
        # Agent
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        self.behaviour_policy = behaviour_policy
        # Memory
        self.max_length = 2
        self.rewards = deque(maxlen=self.max_length)
        self.states = deque(maxlen=self.max_length)
        self.actions = deque(maxlen=self.max_length)
        # Eligibility Trace
        self.trace_decay = trace_decay
        self.traces = np.zeros((n_actions, n_features))
        # Model
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_actions, n_features))
    def start(self, state):
        self.traces.fill(0)
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.append(0)
        self.states.append(state)
    def act(self, state):
        q_values = self.q_values(state)
        return self.behaviour_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        self.actions.append(action)
        self.learn()
        self.rewards.append(reward)
        self.states.append(next_state)
    def learn(self):
        if self.ready():
            self.update_trace()
            self.update_weights()
    def end(self):
        self.actions.append(None)
        if self.memory_length() == 2:
            self.update_trace()
            self.update_weights()
    # Agent
    def returns(self):
        returns = (
            0 if self.actions[-1] is None
            else self.q_value(self.states[-1], self.actions[-1]))
        for reward, state, action in list(zip(self.rewards, self.states, self.actions))[:0:-1]:
            q_values = self.q_values(state)
            q_values[action] = returns
            returns = reward + self.discount_factor * self.target_policy.weighted_sum(q_values)
        return returns
    def td_error(self):
        state, action = self.states[0], self.actions[0]
        return self.returns() - self.q_value(state, action)
    def update_trace(self):
        self.traces *= self.discount_factor * self.trace_decay * self.target_policy
        if self.actions[-1] is not None:
            self.traces[self.actions[-1]] += self.states[-1]
    def update_weights(self):
        self.weights += self.learning_rate * self.td_error() * self.traces
    # Memory
    def ready(self):
        return len(self.rewards) >= self.max_length
    def memory_length(self):
        return len(self.rewards)
    # Model
    def q_value(self, state, action):
        return np.dot(self.weights[action], state)
    def q_values(self, state):
        return np.dot(self.weights, state)

class TreeDutchAgent(Agent):
    def __init__(
        self, n_actions, n_features, discount_factor, 
        target_policy, behaviour_policy, trace_decay, learning_rate):
        # Agent
        self.discount_factor = discount_factor
        self.target_policy = target_policy
        self.behaviour_policy = behaviour_policy
        # Memory
        self.max_length = 2
        self.rewards = deque(maxlen=self.max_length)
        self.states = deque(maxlen=self.max_length)
        self.actions = deque(maxlen=self.max_length)
        # Eligibility Trace
        self.trace_decay = trace_decay
        self.traces = np.zeros((n_actions, n_features))
        # Model
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_actions, n_features))
    def start(self, state):
        self.traces.fill(0)
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.append(0)
        self.states.append(state)
    def act(self, state):
        q_values = self.q_values(state)
        return self.target_policy.choose(q_values)
    def observe(self, action, next_state, reward):
        self.actions.append(action)
        self.learn()
        self.rewards.append(reward)
        self.states.append(next_state)
    def learn(self):
        if self.ready():
            self.update_trace()
            self.update_weights()
    def end(self):
        self.actions.append(None)
        if self.memory_length() == 2:
            self.update_trace()
            self.update_weights()
    # Agent
    def returns(self):
        returns = (
            0 if self.actions[-1] is None
            else self.q_value(self.states[-1], self.actions[-1]))
        for reward, state, action in list(zip(self.rewards, self.states, self.actions))[:0:-1]:
            q_values = self.q_values(state)
            q_values[action] = returns
            returns = reward + self.discount_factor * self.target_policy.weighted_sum(q_values)
        return returns
    def td_error(self):
        state, action = self.states[0], self.actions[0]
        return self.returns() - self.q_value(state, action)
    def update_trace(self):
        self.traces *= self.discount_factor * self.trace_decay
        if self.actions[-1] is not None:
            self.traces[self.actions[-1]] = self.states[-1]
    def update_weights(self):
        self.weights += self.learning_rate * self.td_error() * self.traces
    # Memory
    def ready(self):
        return len(self.rewards) >= self.max_length
    def memory_length(self):
        return len(self.rewards)
    # Model
    def q_value(self, state, action):
        return np.dot(self.weights[action], state)
    def q_values(self, state):
        return np.dot(self.weights, state)