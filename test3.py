from v2.agents import *
from v2.env import GymLinearEnv, PreprocessLinearEnv
from v2.interact import test_env
from v2.features import aggregate
from v2.utils import linear_decay_clip, plot_2d_value_function, compose
import v2.plot as Plot
from pathos.multiprocessing import ProcessingPool
import gym

from pprint import pprint

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    lows = env.observation_space.low
    highs = env.observation_space.high
    env = GymLinearEnv(env)
    env = PreprocessLinearEnv(env, aggregate(lows, highs, 6))
    with ProcessingPool() as p:
        tasks = [
            p.apipe(
                expectedsarsa,
                env.copy(),
                discount=0.99,
                train_ts=int(3e4),
                epsilon=linear_decay_clip(0, 0, 10),
                lr=0.1,
                nstep=1)
            for _ in range(8)]
        histories, qfs = zip(*[task.get() for task in tasks])
        # for qf in qfs:
        #     plot_2d_value_function(
        #         compose(
        #             env.preprocess,
        #             lambda x: max(qf(x))),
        #         lows, highs, invert_z=True,
        #         zlabel='State Value',
        #         xlabel='Position',
        #         ylabel='Velocity')
        Plot.from_histories(histories)