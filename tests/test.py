from v2.returns import discounted
from v2.env import Env
from v2.agents import sarsa, qlearning
from v2.interact import test_env
from v2.features import aggregate
from v2.utils import linear_decay_clip, plot_2d_value_function, compose, argmax
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool
import random
import gym

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    lows = env.observation_space.low
    highs = env.observation_space.high
    env = Env(env, aggregate(lows, highs, [6, 4]))
    def stuff(env, seed):
        env.seed(seed)
        return qlearning(
            env=env,
            discount=0.99,
            train_ts=3000,
            epsilon=linear_decay_clip(0, 0, 100),
            lr=0.1,
            nstep=1)
    env2 = Env(gym.make('MountainCar-v0'), aggregate(lows, highs, [6, 4]))
    with ProcessingPool() as p:
        # qfs = [
        #     p.apply_async(qlearning, dict(
        #         env=env,
        #         discount=0.99,
        #         train_ts=30000,
        #         epsilon=linear_decay_clip(0, 0, 100),
        #         lr=0.1,
        #         nstep=1))
        #     for _ in range(1)]
        qfs = []
        qfs.append(p.apipe(stuff, env, 0))
        qfs.append(p.apipe(stuff, env, 1))
        # qfs = [p.apipe(stuff, env, i) for i in range(2)]
        # qfs = [qf.get() for qf in qfs]
        for qf in qfs:
            qf = qf.get()
            # test_env(
            #     env, e_greedy_const(env.rand_action, 0.0, qf))
            plot_2d_value_function(
                compose(
                    env.preprocess,
                    lambda x: max(qf(x))),
                lows, highs, invert_z=True,
                zlabel='State Value',
                xlabel='Position',
                ylabel='Velocity')
        
        
        
    # qf = qlearning(
    #     env,
    #     discount=0.99,
    #     train_ts=30000,
    #     epsilon=linear_decay_clip(0, 0, 100),
    #     lr=0.1,
    #     nstep=1)
    # test_env(
    #     env, 
    #     e_greedy_const(env.rand_action, 0.0, qf))
    # plot_2d_value_function(
    #     compose(
    #         env.preprocess,
    #         lambda x: max(qf(x))),
    #     lows, highs, invert_z=True,
    #     zlabel='State Value',
    #     xlabel='Position',
    #     ylabel='Velocity')