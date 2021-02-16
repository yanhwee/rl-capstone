import sys
from time import sleep
from IPython.display import clear_output

def render_env(env, delay=0, text=''):
    clear_output(wait=True)
    env.render()
    print(text)
    sleep(delay)

def test_env(env, act, delay=0, ts=sys.maxsize):
    acc_reward = 0
    state = env.reset()
    render_env(env, delay, 0)
    for t in range(ts):
        action = act(state)
        state, reward, done = env.step(action)
        acc_reward += reward
        render_env(env, delay, t)
        if done: break
    print('Total Reward:', acc_reward)