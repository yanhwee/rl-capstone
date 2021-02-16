from v2.env import GymLinearEnv, PreprocessLinearEnv
from v2.utils import compose, linear_decay_clip
from v2.features import fourier, normalise
from v2.agents import qlearning
import v2.plot as Plot
from v2.interact import test_env
import v2.policies as P
import gym

if __name__ == '__main__':
    env = GymLinearEnv(gym.make('MountainCar-v0'))
    lows, highs = env.lows, env.highs
    preprocess = compose(
        normalise(lows, highs),
        fourier([20, 20]))
    env = PreprocessLinearEnv(env, preprocess)
    hist, qf = qlearning(
        env, discount=0.99, train_ts=int(5e5),
        epsilon=linear_decay_clip(0, 0, 1),
        lr=0.001, nstep=1)
    Plot.history(hist)
    Plot.value_function_2d(
        lambda state: max(qf(preprocess(state))),
        lows, highs, intervals=100,
        zlabel='State Value Function',
        xlabel='Position', ylabel='Velocity',
        title='MountainCar-v0', invert_z=True)
    Plot.value_function_2d(
        lambda state: max(qf(preprocess(state))),
        lows, highs, intervals=100,
        zlabel='State Value Function',
        xlabel='Position', ylabel='Velocity',
        title='MountainCar-v0', heatmap=True)
    test_env(env, P.act_e_greedy(env.rand_action, 0, qf))