from v2.memory import Memory
from collections import deque
from tqdm.auto import tqdm

def plain(
    env, max_ts,
    start, act, state_do, action_do, reward_do, end):
    # Assume env.reset() state is not terminal
    # Compile Heavy!
    state, done = None, True
    for ts in tqdm(range(max_ts), disable=False):
        if done:
            state = env.reset()
            start(ts)
        state_do(ts, state)
        action = act(state, ts)
        action_do(ts, action)
        state, reward, done = env.step(action)
        reward_do(ts, reward)
        if done:
            end(ts)
    return dict()

def history(
    env, max_ts,
    start, act, state_do, action_do, reward_do, end):
    eps_rewards = deque()
    acc_reward = 0
    def start2(ts):
        nonlocal acc_reward
        acc_reward = 0
        start(ts)
    def reward_do2(ts, reward):
        nonlocal acc_reward
        acc_reward += reward
        reward_do(ts, reward)
    def end2(ts):
        eps_rewards.append(acc_reward)
        end(ts)
    history = plain(
        env, max_ts, start2, act,
        state_do, action_do, reward_do2, end2)
    history['eps_rewards'] = eps_rewards
    return history

def memory(
    env, max_ts, maxlen,
    start, act, state_do, action_do, reward_do, end):
    states = Memory(maxlen)
    actions = Memory(maxlen)
    rewards = Memory(maxlen)
    def wrap_do(memory, do):
        def wrap(ts, x):
            memory[ts] = x
            do(ts, states, actions, rewards)
        return wrap
    def end2(ts):
        end(ts, states, actions, rewards)
    return history(
        env, max_ts, start, act, 
        wrap_do(states, state_do),
        wrap_do(actions, action_do),
        wrap_do(rewards, reward_do), end2)

def memory_warmup(
    env, max_ts, maxlen, 
    start, act, state_do, action_do, reward_do, end):
    warmup_ts = 0
    def start2(ts):
        nonlocal warmup_ts
        warmup_ts = ts + maxlen - 1
        start(ts)
    def wrap_do(do):
        def wrap(ts, states, actions, rewards):
            if ts >= warmup_ts:
                do(ts, states, actions, rewards)
        return wrap
    return memory(
        env, max_ts, maxlen, start2, act,
        wrap_do(state_do),
        wrap_do(action_do),
        wrap_do(reward_do), end)

def nstep(
    env, max_ts, nstep, 
    start, act, state_do, action_do, reward_do, end):
    start_ts = 0
    def start2(ts):
        nonlocal start_ts
        start_ts = ts
        start(ts)
    def wrap_do(do):
        def wrap(ts, states, actions, rewards):
            do(ts - nstep, ts, states, actions, rewards)
        return wrap
    def end2(ts, states, actions, rewards):
        for t0 in range(max(start_ts, ts - nstep), ts):
            end(t0, ts, states, actions, rewards)
    return memory_warmup(
        env, max_ts, nstep + 1, start2, act,
        wrap_do(state_do),
        wrap_do(action_do),
        wrap_do(reward_do), end2)

# def gpi(
#     env, warmup_t, train_ts,
#     states, actions, rewards,
#     act, state_do, action_do, end):
#     ts = 0
#     while True:
#         # Check
#         warmup_ts = ts + warmup_t
#         if warmup_ts >= train_ts: break
#         # Start
#         state, done = env.reset(), False
#         # Warmup
#         while not done and ts < warmup_ts:
#             states[ts] = state
#             action = act(state, ts)
#             actions[ts] = action
#             state, reward, done = env.step(action)
#             rewards[ts] = reward
#             ts += 1
#         # Train
#         while not done and ts < train_ts:
#             states[ts] = state
#             state_do(ts)
#             action = act(state, ts)
#             actions[ts] = action
#             action_do(ts)
#             state, reward, done = env.step(action)
#             rewards[ts] = reward
#             ts += 1
#         # End
#         if done:
#             end(ts)
#     env.close()