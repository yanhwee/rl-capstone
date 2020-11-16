import gym
from gym import spaces
import numpy as np
from collections import deque
from itertools import count
import sys
import time
from tqdm.notebook import tqdm
from IPython.display import clear_output

class Interact:
    @staticmethod
    def preprocessor(env):
        if isinstance(env.observation_space, spaces.Discrete):
            identity = np.eye(env.observation_space.n)
            return lambda state: identity[state]
        else:
            return lambda state: state
    @staticmethod
    def test(env, agent, delay, limit=sys.maxsize):
        preprocess = Interact.preprocessor(env)
        def render(i):
            clear_output(wait=True)
            env.render()
            print(i)
            time.sleep(delay)
        state = env.reset()
        state = preprocess(state)
        agent.start(state)
        render(0)
        for i in range(limit):
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            state = preprocess(state)
            # agent.observe(action, state, reward)
            render(i)
            if done: break
        env.close()
    @staticmethod
    def train(env, agent, eps):
        preprocess = Interact.preprocessor(env)
        eps_act = [None] * eps
        eps_obs = [None] * eps
        eps_rewards = [None] * eps
        eps_states = [None] * eps
        eps_actions = [None] * eps
        for i in tqdm(range(eps)):
            ep_act = deque()
            ep_obs = deque()
            ep_rewards = deque()
            ep_states = deque()
            ep_actions = deque()

            state = env.reset()
            state = preprocess(state)
            agent.start(state)
            ep_rewards.append(0)
            ep_states.append(state)
            while True:
                one = time.perf_counter()
                action = agent.act(state)
                two = time.perf_counter()
                state, reward, done, info = env.step(action)
                state = preprocess(state)
                three = time.perf_counter()
                agent.observe(action, state, reward)
                four = time.perf_counter()

                ep_act.append(two - one)
                ep_obs.append(four - three)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_states.append(state)
                if done: break
            ep_actions.append(-1)
            agent.end()

            eps_act[i] = ep_act
            eps_obs[i] = ep_obs
            eps_rewards[i] = ep_rewards
            eps_states[i] = ep_states
            eps_actions[i] = ep_actions
        env.close()
        return (eps_act, eps_obs, eps_rewards, eps_states, eps_actions)