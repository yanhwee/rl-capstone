import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import clear_output
import time, sys
from utils import (
    simple_bar, simple_line, simple_scatter, 
    mae, argmax_equal_percent)

class MC:
    def __init__(self, env, gamma, epsilon_start, epsilon_end, alpha_start, alpha_end, max_eps, eval_q_table):
        # Define Constants
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        max_timestep = env._max_episode_steps
        max_length = max_timestep + 1
        epsilon = epsilon_start
        alpha = alpha_start
        epsilon_decay = (epsilon_start - epsilon_end) / max_eps
        alpha_decay = (alpha_start - alpha_end) / max_eps
        # epsilon_decay = (epsilon_end / epsilon_start) ** (1 / (max_eps - 1))
        # alpha_decay = (alpha_end / alpha_start) ** (1 / (max_eps - 1))
        # Monte Carlo
        q_table = np.zeros((n_states, n_actions))
        eps_acc_rewards = [0] * max_eps
        eps_q_loss = [0] * max_eps
        eps_pi_opt = [0] * max_eps
        rewards = [0] * max_length
        states = [0] * max_length
        actions = [0] * max_length
        for i in tqdm(range(max_eps)):
            state = env.reset()
            reward = 0
            acc_reward = 0
            for t in range(max_timestep):
                rewards[t] = reward
                states[t] = state
                action = (
                    np.random.randint(n_actions)
                    if np.random.random() < epsilon
                    else np.argmax(q_table[state]))
                actions[t] = action
                state, reward, done, info = env.step(action)
                acc_reward += reward
                if done: break
            t += 1
            rewards[t] = reward
            states[t] = state
            actions[t] = 0
            returns = reward
            for t in range(t - 1, 0, -1):
                state, action = states[t], actions[t]
                old_value = q_table[state, action]
                q_table[state, action] += alpha * (returns - old_value)
                returns = gamma * returns + rewards[t]
            # Collect Stats
            eps_acc_rewards[i] = acc_reward
            eps_q_loss[i] = mae(q_table, eval_q_table)
            eps_pi_opt[i] = argmax_equal_percent(q_table, eval_q_table)
            epsilon -= epsilon_decay
            alpha -= alpha_decay
            # epsilon *= epsilon_decay
            # alpha *= alpha_decay
        # Save
        self.env, self.gamma, self.epsilon_start, self.epsilon_end, self.epsilon, self.epsilon_decay, self.alpha_start, self.alpha_end, self.alpha, self.alpha_decay, self.max_eps, self.n_states, self.n_actions, self.max_timestep, self.max_length, self.q_table, self.eps_acc_rewards, self.eps_q_loss, self.eps_pi_opt, self.rewards, self.states, self.actions = env, gamma, epsilon_start, epsilon_end, epsilon, epsilon_decay, alpha_start, alpha_end, alpha, alpha_decay, max_eps, n_states, n_actions, max_timestep, max_length, q_table, eps_acc_rewards, eps_q_loss, eps_pi_opt, rewards, states, actions
    def test(self, delay, limit=None):
        env, q_table = self.env, self.q_table
        def render(i):
            clear_output(wait=True)
            env.render()
            print(i)
            time.sleep(delay)
        state = env.reset()
        render(0)
        for i in range(limit if limit else sys.maxsize):
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
            render(i)
            if done: break
        env.close()
    def stats(self, title=None):
        env, gamma, epsilon_start, epsilon_end, epsilon, epsilon_decay, alpha_start, alpha_end, alpha, alpha_decay, max_eps, n_states, n_actions, max_timestep, max_length, q_table, eps_acc_rewards, eps_q_loss, eps_pi_opt, rewards, states, actions = self.env, self.gamma, self.epsilon_start, self.epsilon_end, self.epsilon, self.epsilon_decay, self.alpha_start, self.alpha_end, self.alpha, self.alpha_decay, self.max_eps, self.n_states, self.n_actions, self.max_timestep, self.max_length, self.q_table, self.eps_acc_rewards, self.eps_q_loss, self.eps_pi_opt, self.rewards, self.states, self.actions
        simple_scatter(eps_acc_rewards, 1, 'Total Rewards', 'Episodes', title)
        simple_line(eps_acc_rewards, 1, 'Total Rewards', 'Episodes', title)
        simple_line(eps_q_loss, 1, 'Average State-Action Loss', 'Episodes', title)
        simple_line(eps_pi_opt, 1, '% Optimal Policy', 'Episodes', title)
        print('Final Q Values MAE', eps_q_loss[-1])
        print('Final % Optimal Policy', eps_pi_opt[-1])