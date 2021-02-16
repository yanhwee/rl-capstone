from copy import deepcopy
import numpy as np
import gym

# class Env:
#     def __init__(self, env, preprocess=None):
#         self.env = env
#         self.preprocess = (
#             preprocess if preprocess else lambda x: x)
#         self.n_features = \
#             self.preprocess(env.observation_space.sample()).shape[0]
#         self.n_states = self.n_features
#         self.n_actions = env.action_space.n
#     def rand_action(self):
#         return self.env.action_space.sample()
#     def reset(self):
#         state = self.env.reset()
#         state = self.preprocess(state)
#         return state
#     def step(self, action):
#         state, reward, done, info = self.env.step(action)
#         state = self.preprocess(state)
#         return state, reward, done
#     def render(self):
#         self.env.render()
#     def close(self):
#         self.env.close()
#     def seed(self, seed):
#         self.env.seed(seed)
#         return self
#     def copy(self):
#         return Env(
#             env=gym.make(self.env.unwrapped.spec.id),
#             preprocess=self.preprocess)

class Env:
    def rand_state(self):
        raise NotImplementedError
    def rand_action(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def step(self, action):
        raise NotImplementedError
    def render(self):
        raise NotImplementedError
    def copy(self):
        return deepcopy(self)

class LinearEnv(Env):
    def __init__(self, n_features, n_actions):
        self.n_features = n_features
        self.n_actions = n_actions

class GymLinearEnv(LinearEnv):
    def __init__(self, env):
        self.env = env
        self.n_features = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.highs = env.observation_space.high
        self.lows = env.observation_space.low
    def rand_state(self):
        return self.env.observation_space.sample()
    def rand_action(self):
        return self.env.action_space.sample()
    def reset(self):
        return self.env.reset()
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done
    def render(self):
        self.env.render()
    def copy(self):
        return GymLinearEnv(
            env=gym.make(self.env.unwrapped.spec.id))

class PreprocessLinearEnv(LinearEnv):
    def __init__(self, env, preprocess):
        self.env = env
        self.preprocess = preprocess
        self.n_features = preprocess(env.rand_state()).shape[0]
        self.n_actions = env.n_actions
    def rand_state(self):
        return self.preprocess(self.env.rand_state())
    def rand_action(self):
        return self.env.rand_action()
    def reset(self):
        return self.preprocess(self.env.reset())
    def step(self, action):
        state, reward, done = self.env.step(action)
        state = self.preprocess(state)
        return state, reward, done
    def render(self):
        self.env.render()
    def copy(self):
        return PreprocessLinearEnv(
            self.env.copy(), self.preprocess)

class DiscreteEnv(Env):
    def __init__(self, n_states, n_actions, P):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P

class GymDiscreteEnv(DiscreteEnv):
    def __init__(self, env):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.observation_space.n
        self.P = env.P
    def rand_state(self):
        return self.env.observation_space.sample()
    def rand_action(self):
        return self.env.action_space.sample()
    def reset(self):
        return self.env.reset()
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done
    def render(self):
        self.env.render()
    def copy(self):
        return GymLinearEnv(
            env=gym.make(self.env.unwrapped.spec.id))

# class LinearEnv(Env):
#     def __init__(self, n_features, n_actions, preprocess):
#         self.n_features = n_features
#         self.n_actions = n_actions
#         self.preprocess = preprocess if preprocess else lambda x: x

# class GymLinearEnv(LinearEnv):
#     def __init__(self, env, preprocess):
#         self.env = env
#         self.preprocess = preprocess if preprocess else lambda x: x
#         self.n_features = \
#             self.preprocess(env.observation_space.sample()).shape[0]
#         self.n_actions = env.action_space.n
#     def rand_action(self):
#         return self.env.action_space.sample()
#     def reset(self):
#         state = self.env.reset()
#         state = self.preprocess(state)
#         return state
#     def step(self, action):
#         state, reward, done, info = self.env.step(action)
#         state = self.preprocess(state)
#         return state, reward, done
#     def render(self):
#         self.env.render()
#     def close(self):
#         self.env.close()
#     def copy(self):
#         return GymLinearEnv(
#             env=gym.make(self.env.unwrapped.spec.id),
#             preprocess=self.preprocess)

# class DiscreteEnv(Env):
#     def __init__(self, n_states, n_actions, P):
#         self.n_states = n_states
#         self.n_actions = n_actions
#         self.P = P

# class GymDiscreteEnv(Env):
#     def __init__(self, env):
#         self.env = env
#         self.n_states = env.observation_space.n
#         self.n_actions = env.action_space.n
#         self.P = env.P
#     def rand_action(self):
#         return self.env.action_space.sample()
#     def reset(self):
#         return self.env.reset()
#     def step(self, action):
#         state, reward, done, info = self.env.step(action)
#         return state, reward, done
#     def render(self):
#         self.env.render()
#     def close(self):
#         self.env.close()
#     def copy(self):
#         return GymDiscreteEnv(gym.make(self.env.unwrapped.spec.id))