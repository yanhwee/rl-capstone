import gym
import random
import numpy as np
from bento.client import Client
from bento.sim import Simulation
from bento.example.mountcar import MountainCar, Action, State
from bento.example.specs import Velocity, Position
from IPython.display import clear_output
from tqdm.auto import tqdm

class MountainCarEnv:
    def __init__(self, host='bento.mrzzy.co', port='54242'):
        client = Client(host=host, port=port)
        client.connect(timeout_sec=30)
        if 'mountain_car' in client.list_sims():
            client.remove_sim('mountain_car')
        sim = Simulation.from_def(MountainCar, client)
        self.sim = sim
    def get_state(self):
        car = self.sim.entity(components=[Velocity, Position])
        return np.array([car[Position].x, car[Velocity].x])
    def reset(self):
        try: self.sim.stop()
        except: pass
        self.t = 0
        self.sim.start()
        return self.get_state()
    def step(self, action):
        env = self.sim.entity(components=[Action, State])
        env[Action].accelerate = action
        self.sim.step()
        self.t += 1
        state = self.get_state()
        reward = env[State].reward
        done = env[State].ended if self.t < 200 else True
        return state, reward, done

def tabular_qlearning(
    env, n_states, n_actions, preprocess=lambda x: x,
    alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_table = np.zeros([n_states, n_actions])
    for i in tqdm(range(episodes)):
        state = env.reset()
        state = preprocess(state)
        while True:
            # Act
            action = (
                env.action_space.sample()
                if random.random() < epsilon else 
                np.argmax(q_table[state]))
            # Observe
            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)
            bootstrap = np.max(q_table[next_state])
            backup = reward + gamma * bootstrap
            q_table[state,action] += \
                alpha * (backup - q_table[state,action])
            # Loop
            state = next_state
            if done: break
    policy = lambda state: np.argmax(preprocess(state))
    return policy

def discretize(lows, highs, intervals):
    intervals *= np.ones(len(lows), dtype=int)
    highs = np.nextafter(highs, np.inf)
    widths = (highs - lows) / intervals
    keys = np.cumprod(intervals) // intervals[0]
    clipmax = intervals - 1
    return lambda x: np.dot(keys, np.clip(
        ((x - lows) / widths).astype(int), 0, clipmax))

if __name__ == '__main__':
    # Load Environment
    env = gym.make('MountainCar-v0')
    lows = env.observation_space.low
    highs = env.observation_space.high
    n_actions = env.action_space.n
    print(f'Number of Features: {len(lows)}')
    print(f'Position: [{lows[0]}, {highs[0]}]')
    print(f'Velocity: [{lows[1]}, {highs[1]}]')
    print(f'Number of Actions: {n_actions}')
    # Preprocess
    intervals = 6
    preprocess = discretize(lows, highs, intervals)
    n_states = intervals ** 2
    # Run
    policy = tabular_qlearning(
        env, n_states, n_actions, preprocess=preprocess,
        alpha=0.1, gamma=0.99, epsilon=0.0, episodes=1000)
    # Test
    state = env.reset()
    while True:
        # Act
        action = policy(state)
        # Observe
        state, reward, done, info = env.step(action)
        # Render
        clear_output(wait=True)
        env.render()
        # Loop
        if done: break