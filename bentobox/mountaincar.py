from v2.env import LinearEnv
import numpy as np
import gym
from bento.client import Client
from bento.sim import Simulation
from bento.example.mountcar import MountainCar, Action, State
from bento.example.specs import Velocity, Position

ENGINE_HOST = 'bento.mrzzy.co'
ENGINE_PORT = '54242'

class BentoMountainCarEnv(LinearEnv):
    def __init__(self, host=ENGINE_HOST, port=ENGINE_PORT):
        # Bento
        client = Client(host=host, port=port)
        status = client.connect(timeout_sec=30)
        print('Client Status:', status)
        if 'mountain_car' in client.list_sims():
            client.remove_sim('mountain_car')
        sim = Simulation.from_def(MountainCar, client)
        self.client, self.sim = client, sim
        # Gym
        env = gym.make('MountainCar-v0').unwrapped
        self.n_features = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.highs = env.observation_space.high
        self.lows = env.observation_space.low
        self.env = env
        # Timestep
        self.t = 0
    def rand_state(self):
        return self.env.observation_space.sample()
    def rand_action(self):
        return self.env.action_space.sample()
    def reset(self):
        try:    self.sim.stop()
        except: pass
        self.t = 0
        self.sim.start()
        return self.get_state()
    def step(self, action):
        self.t += 1
        self.move_car(action)
        state, reward, done = self.get_state(), self.get_reward(), self.get_done()
        if self.t >= 200: done = True
        return state, reward, done
    def render(self):
        self.env.state = self.get_state()
        self.env.render()
    ## Bento Helper
    def get_entities(self):
        car = self.sim.entity(components=[Velocity, Position])
        env = self.sim.entity(components=[Action, State])
        return car, env
    def get_state(self):
        car, env = self.get_entities()
        return np.array([car[Position].x, car[Velocity].x])
    def get_reward(self):
        car, env = self.get_entities()
        return env[State].reward
    def get_done(self):
        car, env = self.get_entities()
        return env[State].ended
    def move_car(self, action):
        car, env = self.get_entities()
        env[Action].accelerate = action
        self.sim.step()