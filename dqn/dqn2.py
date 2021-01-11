from typing import Callable
from collections import deque
from functools import reduce
from copy import deepcopy
import numpy as np
from gym import Env
from tqdm.auto import tqdm
import torch

State = np.ndarray
Action = np.ndarray

class Model:
    def q_values(self, state: State) -> [float]:
        raise NotImplementedError
    def update(
        self, states: [State], actions: [Action], 
        targets: [float]) -> float:
        raise NotImplementedError
    def copy(self):
        raise NotImplementedError

class AggregateModel(Model):
    def __init__(self, lows, highs, buckets, n_actions, lr):
        assert(len(lows) == len(highs))
        highs = np.nextafter(highs, np.inf)
        size = buckets ** len(lows)
        self.lows = lows
        self.highs = highs
        self.buckets = buckets
        self.widths = (highs - lows) / buckets
        self.weights = np.zeros((size, n_actions))
        self.lr = lr
    def q_values(self, state):
        index = reduce(
            lambda x, y: self.buckets * x + y, 
            ((state - self.lows) // self.widths).astype(int))
        return self.weights[index]
    def update(self, states, actions, targets):
        squared_error = 0
        for state, action, target in zip(states, actions, targets):
            q_values = self.q_values(state)
            td_error = target - q_values[action]
            q_values[action] += self.lr * td_error
            squared_error += td_error ** 2
        return squared_error / len(states)
    def copy(self):
        return deepcopy(self)

def dqn(
    env: Env,
    gamma: int,
    epsilon: Callable[[int], float],
    model: Model,
    capacity: int,
    warmup_steps: int, train_steps: int, 
    update_steps: int, batch_size: int) -> [Model, dict]:
    dataset = deque(maxlen=capacity) # SARDS
    model = model.copy()
    target_model = model.copy()
    losses = deque()
    rewards = deque()
    def act(t, state):
        # pylint: disable=maybe-no-member
        return (
            env.action_space.sample()
            if np.random.random() < epsilon(t)
            else np.argmax(model.q_values(state)))
    def learn(t):
        states = [None] * batch_size
        actions = [None] * batch_size
        targets = [None] * batch_size
        for i, j in enumerate(np.random.randint(len(dataset), size=batch_size)):
            state, action, reward, done, next_state = dataset[j]
            target = reward if done else \
                reward + gamma * np.amax(target_model.q_values(next_state))
            states[i] = state
            actions[i] = action
            targets[i] = target
        loss = model.update(states, actions, targets)
        losses.append(loss)
    done = True
    for t in tqdm(range(1, warmup_steps + 1), desc='Warmup'):
        if done: state = env.reset()
        action = act(t, state)
        next_state, reward, done, _ = env.step(action)
        dataset.append((state, action, reward, done, next_state))
        state = next_state
    done = True
    for t in tqdm(range(1, train_steps + 1), desc='Train'):
        if done:
            state = env.reset()
            rewards.append(0)
        action = act(t, state)
        next_state, reward, done, _ = env.step(action)
        dataset.append((state, action, reward, done, next_state))
        state = next_state
        rewards[-1] += reward
        learn(t)
        if t % update_steps == 0:
            target_model = model.copy()
    if not done: rewards.pop()
    env.close()
    history = {
        'eps_rewards': rewards,
        'losses': losses }
    return model, history