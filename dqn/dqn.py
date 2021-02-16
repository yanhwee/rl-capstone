import sys
import numpy as np
from collections import deque
from tqdm.auto import tqdm
from v1.utils import render_env, simple_scatter, simple_line, key_mapper, plot_2d_value_function, normaliser, test_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class DQN:
    def __init__(self, env, gamma, epsilon, model, capacity, warmup_steps, train_steps, update_steps, sample_size):
        dataset = deque(maxlen=capacity + 1) # SARD

        def act(t):
            # pylint: disable=maybe-no-member
            return (
                env.action_space.sample()
                if np.random.random() < epsilon(t)
                else np.argmax(model.q_values(state)))
        
        losses = deque()
        def learn(t):
            if t % update_steps == 0:
                states = [None] * sample_size
                actions = [None] * sample_size
                targets = [None] * sample_size
                for i in range(sample_size):
                    j = np.random.randint(len(dataset) - 1)
                    state, action, reward, done = dataset[j]
                    target = reward
                    if not done:
                        next_state = dataset[j + 1][0]
                        target += gamma * np.amax(model.q_values(next_state))
                    states[i] = state
                    actions[i] = action
                    targets[i] = target
                loss = model.update(states, actions, targets)
                losses.append(loss)

        done = True
        for t in tqdm(range(warmup_steps), desc='Warmup'):
            if done: state = env.reset()
            action = act(t)
            next_state, reward, done, _ = env.step(action)
            dataset.append((state, action, reward, done))
            state = next_state

        done = True
        rewards = deque()
        for t in tqdm(range(train_steps), desc='Train'):
            if done:
                state = env.reset()
                rewards.append(0)
            action = act(t)
            next_state, reward, done, _ = env.step(action)
            dataset.append((state, action, reward, done))
            state = next_state
            learn(t)
            rewards[-1] += reward
        if not done: rewards.pop()

        env.close()
        self.env, self.model = env, model
        self.rewards, self.losses = rewards, losses
        self.train_steps, self.update_steps = train_steps, update_steps
        
    def test(self, delay=0, ts=sys.maxsize):
        env, model = self.env, self.model
        test_env(env, delay, ts, action_func=lambda state: \
            np.argmax(model.q_values(state)))

    def plot_2d_value_function(self, intervals=10, labels=None, title=None, invert_v=False, anim=False):
        env, model = self.env, self.model
        plot_2d_value_function(
            vf=lambda states: [np.amax(model.q_values(state)) for state in states],
            env=env,
            intervals=intervals,
            labels=labels,
            title=title,
            invert_v=invert_v,
            anim=anim)

    def plot_stats(self, title=None):
        rewards, losses = self.rewards, self.losses
        train_steps, update_steps = self.train_steps, self.update_steps
        simple_scatter(rewards, 1, 'Total Rewards', 'Episodes', title=title)
        simple_line(losses, range(0, train_steps, update_steps), 'Loss', 'Train Steps', title=title)

class Model:
    def q_values(self, state):
        raise NotImplementedError
    # def update(self, state, action, target):
    #     raise NotImplementedError
    def update(self, states, actions, targets):
        raise NotImplementedError

class AggregateModel(Model):
    def __init__(self, lows, highs, buckets, n_actions, lr):
        highs = np.nextafter(highs, np.inf)
        self.lows = lows
        self.highs = highs
        self.buckets = buckets
        self.widths = (highs - lows) / buckets
        self.key_map = key_mapper([buckets] * len(lows))
        self.weights = np.zeros((buckets ** len(lows), n_actions))
        self.lr = lr
    def q_values(self, state):
        index = self.key_map((state - self.lows) // self.widths)
        return self.weights[index]
    def update(self, states, actions, targets):
        acc_td_error = 0
        for state, action, target in zip(states, actions, targets):
            q_values = self.q_values(state)
            td_error = target - q_values[action]
            q_values[action] += self.lr * td_error
            acc_td_error += td_error ** 2
        return acc_td_error / len(states)

class BaseModel:
    @staticmethod
    def q_values(model, state):
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float) # pylint: disable=not-callable
            return model.forward(x).cpu().numpy()
    @staticmethod
    def update(model, opt, batch_size, states, actions, targets):
        # pylint: disable=not-callable
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.float)
        ds = TensorDataset(states, actions, targets)
        dl = DataLoader(ds, batch_size=batch_size)
        acc_loss = 0
        for batch_states, batch_actions, batch_targets in dl:
            batch_q_values1 = model.forward(batch_states)
            batch_q_values2 = batch_q_values1.detach().clone()
            batch_q_values2[range(len(batch_q_values2)), batch_actions] = batch_targets
            loss = F.mse_loss(batch_q_values1, batch_q_values2)
            loss.backward()
            opt.step()
            opt.zero_grad()
            acc_loss += loss.item() * len(batch_states)
        return acc_loss / len(ds)

class SampleModel(Model, nn.Module):
    def __init__(self, lows, highs, n_actions, lr, batch_size):
        super().__init__()
        n_features = len(lows)
        self.norm = normaliser(torch.tensor(lows), torch.tensor(highs))
        self.fc1 = nn.Linear(n_features, 16)
        # self.fc2 = nn.Linear(16, 16)
        # self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, n_actions)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.batch_size = batch_size
        self
    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def q_values(self, state):
        return BaseModel.q_values(self, state)
    def update(self, states, actions, targets):
        return BaseModel.update(self, self.opt, self.batch_size, states, actions, targets)

class BadModel(Model, nn.Module):
    def __init__(self, lows, highs, n_actions, lr, batch_size):
        super().__init__()
        n_features = len(lows)
        self.norm = normaliser(lows, highs)
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, n_actions)
        self.opt = optim.SGD(self.parameters(), lr=lr)
        self.batch_size = batch_size
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    def q_values(self, state):
        return BaseModel.q_values(self, state)
    def update(self, states, actions, targets):
        return BaseModel.update(self, self.opt, self.batch_size, states, actions, targets)