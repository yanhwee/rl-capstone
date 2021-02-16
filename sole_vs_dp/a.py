from v2.sole import sole_step
from v2.dp import dp_step
from v2.env import GymDiscreteEnv
from v2.utils import argmax
from v2.interact import test_env
from collections import deque
import time
import gym

def qf_policy(n_states, qf):
    return [argmax(qf(state)) for state in range(n_states)]

def till_optimal_policy(env, discount, algo):
    qf, step = algo(env, discount)
    old_policies = deque(maxlen=2)
    old_policies.append(qf_policy(env.n_states, qf))
    while True:
        step()
        new_policy = qf_policy(env.n_states, qf)
        if new_policy in old_policies: break
        old_policies.append(new_policy)
    return new_policy

def time_function(f, n):
    start = time.perf_counter()
    for _ in range(n): f()
    end = time.perf_counter()
    duration = end - start
    average = duration / n
    return average

def time_till_optimal_policy(env, discount, algo, n):
    return time_function(
        lambda: till_optimal_policy(env, discount, algo), n)

if __name__ == '__main__':
    greedy_sole_step = lambda env, discount: \
        sole_step(env, discount, stochastic=False)
    discount = 0.9
    n = 10
    taxi = GymDiscreteEnv(gym.make('Taxi-v3'))
    lake = GymDiscreteEnv(gym.make('FrozenLake8x8-v0'))
    print(taxi.n_states)
    print(lake.n_states)
    # taxi_sole_time = time_till_optimal_policy(
    #     taxi, discount, greedy_sole_step, n)
    # print(taxi_sole_time)
    # taxi_dp_time = time_till_optimal_policy(
    #     taxi, discount, dp_step, n)
    # print(taxi_dp_time)
    lake_sole_time = time_till_optimal_policy(
        lake, discount, greedy_sole_step, n)
    print(lake_sole_time)
    lake_dp_time = time_till_optimal_policy(
        lake, discount, dp_step, n)
    print(lake_dp_time)
    
    # print(taxi_sole_time)
    # print(taxi_dp_time)
    # print(lake_sole_time)
    # print(lake_dp_time)