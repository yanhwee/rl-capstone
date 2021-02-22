import gym
import numpy as np
from collections import deque

class Interact:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = True
        self.info = None
    
    def step(self):
        if self.done:
            self.agent.reset()
            self.next_state = self.env.reset()
        self.state = self.next_state
        self.action = self.agent.act(self.state)
        self.next_state, self.reward, self.done, self.info = self.env.step(self.action)
        self.agent.learn(self.state, self.action, self.reward, self.next_state)
        if self.done:
            self.agent.finish()

class Agent:
    def act(self, state):
        raise NotImplementedError
    def learn(self, state, action, reward, next_state):
        raise NotImplementedError
    def finish(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

class QLearning(Agent):
    def __init__(self, state_space, action_space, gamma, alpha, epsilon):
        self.state_space = state_space
        self.action_space = action_space
        self.q_values = np.zeros([state_space, action_space])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_values[state])
    def learn(self, state, action, reward, next_state):
        old_value = self.q_values[state, action]
        bootstrap = np.max(self.q_values[next_state])
        target = reward + self.gamma * bootstrap
        td_error = target - old_value
        self.q_values[state, action] += self.alpha * td_error
    def finish(self):
        pass
    def reset(self):
        pass

class Sarsa(Agent):
    def __init__(self, state_space, action_space, gamma, alpha, epsilon):
        self.state_space = state_space
        self.action_space = action_space
        self.q_values = np.zeros([state_space, action_space])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_values[state])
    def learn(self, state, action, reward, next_state):
        if self.prev_state is not None:
            old_value = self.q_values[self.prev_state, self.prev_action]
            bootstrap = self.q_values[state, action]
            target = reward + self.gamma * bootstrap
            td_error = target - old_value
            self.q_values[self.prev_state, self.prev_action] += self.alpha * td_error
        self.prev_state = state
        self.prev_action = action
        self.prev_reward = reward
    def finish(self):
        if self.prev_state is not None:
            old_value = self.q_values[self.prev_state, self.prev_action]
            target = self.prev_reward
            td_error = target - old_value
            self.q_values[self.prev_state, self.prev_action] += self.alpha * td_error
    def reset(self):
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

class TD(Agent):
    def __init__(self, state_space, gamma, alpha):
        self.state_space = state_space
        self.values = np.zeros([state_space])
        self.gamma = gamma
        self.alpha = alpha
    def learn(self, state, action, reward, next_state):
        old_value = self.values[state]
        bootstrap = self.values[next_state]
        target = reward + self.gamma * bootstrap
        td_error = target - old_value
        self.values[state] += self.alpha * td_error
    def finish(self):
        pass
    def reset(self):
        pass

class TDNStep(Agent):
    def __init__(self, state_space, gamma, alpha, n_step):
        self.state_space = state_space
        self.values = np.zeros([state_space])
        self.gamma = gamma
        self.alpha = alpha
        self.n_step = n_step
        self.prev_states = deque(maxlen=n_step)
        self.prev_rewards = deque(maxlen=n_step)
    def learn(self, state, action, reward, next_state):
        self.prev_states.append(state)
        self.prev_rewards.append(reward)
        if len(self.prev_states) == self.n_step:
            prev_state = self.prev_states[0]
            old_value = self.values[prev_state]
            returns = self.values[next_state]
            for reward in reversed(self.prev_rewards):
                returns = self.gamma * returns + reward
            td_error = returns - old_value
            self.values[prev_state] += self.alpha * td_error
            self.prev_states.popleft()
            self.prev_rewards.popleft()
    def finish(self):
        while self.prev_states:
            prev_state = self.prev_states[0]
            old_value = self.values[prev_state]
            returns = 0
            for reward in reversed(self.prev_rewards):
                returns = self.gamma * returns + reward
            td_error = returns - old_value
            self.values[prev_state] += self.alpha * td_error
            self.prev_states.popleft()
            self.prev_rewards.popleft()
    def reset(self):
        self.prev_states.clear()
        self.prev_rewards.clear()

class MonteCarlo(Agent):
    def __init__(self, state_space, gamma, alpha):
        self.state_space = state_space
        self.values = np.zeros([state_space])
        self.gamma = gamma
        self.alpha = alpha
        self.prev_states = deque()
        self.prev_rewards = deque()
    def learn(self, state, action, reward, next_state):
        self.prev_states.append(state)
        self.prev_rewards.append(reward)
    def finish(self):
        while self.prev_states:
            prev_state = self.prev_states[0]
            old_value = self.values[prev_state]
            returns = 0
            for reward in reversed(self.prev_rewards):
                returns = self.gamma * returns + reward
            td_error = returns - old_value
            self.values[prev_state] += self.alpha * td_error
            self.prev_states.popleft()
            self.prev_rewards.popleft()
    def reset(self):
        self.prev_states.clear()
        self.prev_rewards.clear()

class SarsaNStep(Agent):
    def __init__(self, state_space, action_space, gamma, alpha, epsilon, n_step):
        self.state_space = state_space
        self.action_space = action_space
        self.q_values = np.zeros([state_space, action_space])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_step = n_step
        self.prev_states = deque(maxlen=n_step)
        self.prev_actions = deque(maxlen=n_step)
        self.prev_rewards = deque(maxlen=n_step)
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_values[state])
    def learn(self, state, action, reward, next_state):
        if len(self.prev_states) == self.n_step:
            prev_state = self.prev_states[0]
            prev_action = self.prev_actions[0]
            old_value = self.q_values[prev_state, prev_action]
            returns = self.q_values[state, action]
            for reward in reversed(self.prev_rewards):
                returns = self.gamma * returns + reward
            td_error = returns - old_value
            self.q_values[prev_state, prev_action] += self.alpha * td_error
            self.prev_states.popleft()
            self.prev_actions.popleft()
            self.prev_rewards.popleft()
        self.prev_states.append(state)
        self.prev_actions.append(action)
        self.prev_rewards.append(reward)
    def finish(self):
        while self.prev_states:
            prev_state = self.prev_states[0]
            prev_action = self.prev_actions[0]
            old_value = self.q_values[prev_state, prev_action]
            returns = 0
            for reward in reversed(self.prev_rewards):
                returns = self.gamma * returns + reward
            td_error = returns - old_value
            self.q_values[prev_state, prev_action] += self.alpha * td_error
            self.prev_states.popleft()
            self.prev_actions.popleft()
            self.prev_rewards.popleft()
    def reset(self):
        self.prev_states.clear()
        self.prev_actions.clear()
        self.prev_rewards.clear()

class QMonteCarlo(Agent):
    def __init__(self, state_space, action_space, gamma, alpha, epsilon):
        self.state_space = state_space
        self.action_space = action_space
        self.q_values = np.zeros([state_space, action_space])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.prev_states = deque()
        self.prev_actions = deque()
        self.prev_rewards = deque()
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_values[state])            
    def learn(self, state, action, reward, next_state):
        self.prev_states.append(state)
        self.prev_actions.append(action)
        self.prev_rewards.append(reward)
    def finish(self):
        while self.prev_states:
            prev_state = self.prev_states[0]
            prev_action = self.prev_actions[0]
            old_value = self.q_values[prev_state, prev_action]
            returns = 0
            for reward in reversed(self.prev_rewards):
                returns = self.gamma * returns + reward
            td_error = returns - old_value
            self.q_values[prev_state, prev_action] += self.alpha * td_error
            self.prev_states.popleft()
            self.prev_actions.popleft()
            self.prev_rewards.popleft()
    def reset(self):
        self.prev_states.clear()
        self.prev_actions.clear()
        self.prev_rewards.clear()

class ExpectedSarsa(Agent):
    def __init__(self, state_space, action_space, gamma, alpha, epsilon):
        self.state_space = state_space
        self.action_space = action_space
        self.q_values = np.zeros([state_space, action_space])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_values[state])
    def action_probabilities(self, state):
        probabilities = np.full([self.action_space], self.epsilon / self.action_space)
        probabilities[np.argmax(self.q_values[state])] += 1 - self.epsilon
        return probabilities
    def learn(self, state, action, reward, next_state):
        old_value = self.q_values[state, action]
        probabilities = self.action_probabilities(next_state)
        bootstrap = np.dot(self.q_values[next_state], probabilities)
        target = reward + self.gamma * bootstrap
        td_error = target - old_value
        self.q_values[state, action] += self.alpha * td_error
    def finish(self):
        pass

class TreeBackup(Agent):
    def __init__(self, state_space, action_space, gamma, alpha, epsilon, n_step):
        self.state_space = state_space
        self.action_space = action_space
        self.q_values = np.zeros([state_space, action_space])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_step = n_step
        self.prev_states = deque(maxlen=n_step)
        self.prev_actions = deque(maxlen=n_step)
        self.prev_rewards = deque(maxlen=n_step)
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_values[state])
    def action_probabilities(self, state):
        probabilities = np.full([self.action_space], self.epsilon / self.action_space)
        probabilities[np.argmax(self.q_values[state])] += 1 - self.epsilon
        return probabilities
    def learn(self, state, action, reward, next_state):
        self.prev_states.append(state)
        self.prev_actions.append(action)
        self.prev_rewards.append(reward)
        if len(self.prev_states) == self.n_step:
            prev_state = self.prev_states.popleft()
            prev_action = self.prev_actions.popleft()
            prev_reward = self.prev_rewards.popleft()
            old_value = self.q_values[prev_state, prev_action]
            probabilities = self.action_probabilities(next_state)
            returns = np.dot(self.q_values[next_state], probabilities)
            for state, action, reward in zip(reversed(self.prev_states), reversed(self.prev_actions), reversed(self.prev_rewards)):
                probabilities = self.action_probabilities(state)
                values = self.q_values[state].copy()
                values[action] = reward + self.gamma * returns
                returns = np.dot(values, probabilities)
            returns = prev_reward + self.gamma * returns
            td_error = returns - old_value
            self.q_values[prev_state, prev_action] += self.alpha * td_error
    def finish(self):
        while self.prev_states:
            prev_state = self.prev_states.popleft()
            prev_action = self.prev_actions.popleft()
            prev_reward = self.prev_rewards.popleft()
            old_value = self.q_values[prev_state, prev_action]
            returns = 0
            for state, action, reward in zip(reversed(self.prev_states), reversed(self.prev_actions), reversed(self.prev_rewards)):
                probabilities = self.action_probabilities(state)
                values = self.q_values[state].copy()
                values[action] = reward + self.gamma * returns
                returns = np.dot(values, probabilities)
            returns = prev_reward + self.gamma * returns
            td_error = returns - old_value
            self.q_values[prev_state, prev_action] += self.alpha * td_error
    def reset(self):
        self.prev_states.clear()
        self.prev_actions.clear()
        self.prev_rewards.clear()

class SarsaLambda(Agent):
    def __init__(self, state_space, action_space, gamma, alpha, epsilon, n_step, _lambda):
        self.state_space = state_space
        self.action_space = action_space
        self.q_values = np.zeros([state_space, action_space])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_step = n_step
        self._lambda = _lambda
        self.prev_states = deque(maxlen=n_step)
        self.prev_actions = deque(maxlen=n_step)
        self.prev_reward = None
        self.prev_traces = deque(maxlen=n_step)
    def learn(self, state, action, reward, next_state):
        if self.prev_reward:
            prev_state = self.prev_states[-1]
            prev_action = self.prev_actions[-1]
            old_value = self.q_values[prev_state, prev_action]
            bootstrap = self.q_values[state, action]
            target = self.prev_reward + self.gamma * bootstrap
            td_error = target - old_value
            for state, action, trace in zip(reversed(self.prev_states), reversed(self.prev_actions), reversed(self.prev_traces)):
                self.q_values[state, action] += trace * td_error
            for _ in range(len(self.prev_traces)):
                self.prev_traces.append(self.gamma * self._lambda * self.prev_traces.popleft())
        self.prev_states.append(state)
        self.prev_actions.append(action)
        self.prev_reward = reward
        self.prev_traces.append(1)
    def finish(self):
        if self.prev_reward:
            prev_state = self.prev_states[-1]
            prev_action = self.prev_actions[-1]
            old_value = self.q_values[prev_state, prev_action]
            td_error = self.prev_reward - old_value
            for state, action, trace in zip(reversed(self.prev_states), reversed(self.prev_actions), reversed(self.prev_traces)):
                self.q_values[state, action] += trace * td_error
    def reset(self):
        self.prev_states.clear()
        self.prev_actions.clear()
        self.prev_reward = None
        self.prev_traces.clear()