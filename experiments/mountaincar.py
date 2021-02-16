from v2.env import GymLinearEnv, PreprocessLinearEnv
from v2.features import aggregate, tab2lin
from v2.agents import qlearning
from v2.utils import compose, linear_decay_clip
import v2.plot as Plot
from v2.interact import test_env
import v2.policies as P
import numpy as np
import gym

if __name__ == '__main__':
    env = GymLinearEnv(gym.make('MountainCar-v0'))
    lows, highs = env.lows, env.highs
    print(f'Number of Features: {env.n_features}')
    print(f'Position: [{lows[0]}, {highs[0]}]')
    print(f'Velocity: [{lows[1]}, {highs[1]}]')
    print(f'Number of Actions: {env.n_actions}')
    intervals = [6, 6]
    preprocess = compose(
        aggregate(lows, highs, intervals),
        tab2lin(np.prod(intervals)))
    env = PreprocessLinearEnv(env, preprocess)
    hist, qf = qlearning(
        env, discount=0.99, train_ts=int(5e5),
        epsilon=linear_decay_clip(0, 0, 1),
        lr=0.1, nstep=1)
    Plot.history(hist)
    Plot.value_function_2d(
        lambda state: max(qf(preprocess(state))),
        lows, highs, zlabel='State Value Function',
        xlabel='Position', ylabel='Velocity',
        title='MountainCar-v0', invert_z=True)
    Plot.value_function_2d(
        lambda state: max(qf(preprocess(state))),
        lows, highs, zlabel='State Value Function',
        xlabel='Position', ylabel='Velocity',
        title='MountainCar-v0', heatmap=True)
    test_env(env, P.act_e_greedy(env.rand_action, 0, qf))