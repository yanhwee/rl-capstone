# state = env.reset()

# states[ts] = state
# state_learn()
# action = act(state, ts)
# actions[ts] = action
# action_learn()
# state, reward, done = env.step(action)
# rewards[ts] = reward
# if done: break
# ts += 1

# states[ts] = state
# action = act(state, ts)
# actions[ts] = action
# state, reward, done = env.step(action)
# rewards[ts] = reward
# if done: break
# ts += 1

# for _ in range(n_step):
#     rewards[ts] = reward
#     states[ts] = state
#     if done: break
#     action = act(state, ts)
#     actions[ts] = action
#     ts += 1
#     state, reward, done = env.step(action)
# while True:
#     rewards.append(reward)
#     states.append(state)
#     if done: break
#     action = act(state, ts)
#     actions.append(action)
#     ts += 1
#     state, reward, done = env.step(action)

# state, reward, done = env.reset()
# rewards.append(reward)
# states.append(state)
# action = act(state, ts)
# actions.append(action)
# if done: break
# ts += 1

# for ts in range(ts, ts + 1):
#     state, reward, done = env.reset(), 0, False
#     rewards[ts] = reward
#     states[ts] = state
#     if done: break
#     action = act(state, ts)
#     actions[ts] = action
# # Warmup
# for ts in range(ts, ts + n_step):
#     state, reward, done = env.step(action)
#     rewards[ts] = reward
#     states[ts] = state
#     if done: break
#     action = act(state, ts)
#     actions[ts] = action
# # Train
# for ts in range(ts, max_ts):
#     state, reward, done = env.step(action)
#     rewards[ts] = reward
#     states[ts] = state
#     state_learn()
#     if done: break
#     action = act(state, ts)
#     actions[ts] = action
#     action_learn()

def sarsa(env, discount, epsilon, lr, n_step, train_ts):
    # Env
    n_features = env.n_features
    n_actions = env.n_actions
    rand_action = lambda: randrange(n_actions)
    # Memory
    maxlen = n_step + 1
    states = Memory(maxlen)
    actions = Memory(maxlen)
    rewards = Memory(maxlen)
    # Linear Model
    weights = np.zeros((n_actions, n_features))
    q_values = lambda state: np.dot(weights, state)
    q_value = lambda state, action: np.dot(weights[action], state)
    def update_q_value(state, action, target):
        q = weights[action, state]
        q += lr * (target - q)
        weights[action, state] = q
    # Functions
    act = lambda state, ts: (
        rand_action()
        if randbool(epsilon(ts))
        else argmax(q_values(state)))
    def state_backup(ts1, ts2):
        pass
    def action_backup(ts1, ts2):
        update_q_value(
            states[ts1], actions[ts1],
            G.sarsa(
                discount, ts1, ts2,
                states, actions, rewards,
                q_value))
    def end_backup(ts1, ts2):
        update_q_value(
            states[ts1], actions[ts1],
            G.sarsa_terminal(
                discount, ts1, ts2, rewards))
    # GPI
    ts = 0
    while True:
        # Check
        warmup_ts = ts + n_step
        if warmup_ts >= train_ts: break
        # Start
        state, done = env.reset(), False
        # Warmup
        while not done and ts < warmup_ts:
            states[ts] = state
            action = act(state, ts)
            actions[ts] = action
            state, reward, done = env.step(action)
            rewards[ts] = reward
            ts += 1
        # Train
        while not done and ts < train_ts:
            states[ts] = state
            state_backup(ts - n_step, ts)
            action = act(state, ts)
            actions[ts] = action
            action_backup(ts - n_step, ts)
            state, reward, done = env.step(action)
            rewards[ts] = reward
            ts += 1
        # End
        if done:
            for t in range(ts - n_step, ts):
                end_backup(t, ts)
    env.close()


def sample_0(states, actions, q_value):
    return lambda ts: q_value(states[ts], actions[ts])

def sample(rewards):
    return lambda ts, g: rewards[ts]

def expected_0(states, q_values, policy):
    return lambda ts: np.dot(q_values(states[ts]), policy(states[ts]))

def expected(states, actions, q_values, policy):
    def returns(ts, g):
        qs = q_values(states[ts])
        qs[actions[ts]] = g
        return np.dot(policy(states[ts]), qs)
    return returns

def sample(ts, states, actions, rewards, q_value):
    rewards[ts-1] + q_value(states[ts], actions[ts])

def expected(ts, states, rewards, q_values, policy):
    rewards[ts-1] + np.dot(
        q_values(states[ts]),
        policy(states[ts]))

def expected_sarsa(discount, ts1, ts2, states, actions, rewards, q_values, policy):
    return rewards[ts1] + discount * expected_sarsa(discount, ts1 + 1, ts2, )
    g = rewards[ts2]
    qs = q_values(states[ts2])
    qs[actions[ts2]] = g
    g = rewards[ts2 - 1] + \
        np.dot(qs, policy(states[ts2]))
    

    for ts in range(ts2, ts1, -1):
        qs = q_values(states[ts])
        qs[actions[ts]]
        g = returns(ts, g)


def discounted(discount, ts1, ts2, rewards, return0, returns):
    g = return0(ts2)
    for ts in range(ts2 - 1, ts1, -1):
        g = rewards[ts] + discount * returns(ts)
    return g

def discounted_terminal(discount, ts1, ts2, return0, returns):
    g = 0
    for ts in range(ts2, ts1, -1):
        g = rewards


import v2.rewards as R
import numpy as np

def sarsa(discount, ts1, ts2, states, actions, rewards, q_value):
    t0, t1 = ts1, ts1 + 1
    return rewards[t0] + discount * (
        sarsa(discount, t1, ts2, states, actions, rewards, q_value)
        if t1 < ts2 else
        q_value(states[ts2], actions[ts2]))

def sarsa_terminal(discount, ts1, ts2, rewards):
    t0, t1 = ts1, ts1 + 1
    return rewards[t0] + discount * (
        sarsa_terminal(discount, t1, ts2, rewards)
        if t1 < ts2 else rewards[ts2])

def tree(discount, ts1, ts2, states, actions, rewards, q_values, policy):
    t0, t1 = ts1, ts1 + 1
    return rewards[t0] + discount * np.dot(
        policy(states[t1]), (
            array_walrus(
                q_values(states[t1]), actions[t1], tree(
                    discount, t1, ts2, states, actions, rewards, 
                    q_values, policy))
            if t1 < ts2 else
            q_values(states[t1])))

def discounted(discount, ts1, ts2, reward0, reward):
    g = reward0(ts2)
    for ts in range(ts2 - 1, ts1, -1):
        g = discount * g + reward(ts)
    return g

def discounted_terminal(discount, ts1, ts2, reward):
    g = 0
    for ts in range(ts2, ts1, -1):
        g = discount * g + reward(ts)
    return g

def sarsa(discount, ts1, ts2, states, actions, rewards, q_value):
    return discounted(
        discount, ts1, ts2,
        R.sample_0(states, actions, q_value),
        R.sample(rewards))

def sarsa_terminal(discount, ts1, ts2, rewards):
    return discounted_terminal(
        discount, ts1, ts2,
        R.sample(rewards))

def expected_sarsa(discount, ts1, ts2, states, rewards, q_values, policy):
    return discounted(
        discount, ts1, ts2,
        R.expected_0(states, q_values, policy),
        R.sample(rewards))

def expected_sarsa_terminal(discount, ts1, ts2, states, rewards, q_values, policy):
    return sarsa_terminal(discount, ts1, ts2, rewards)

def tree(discount, ts1, ts2, states, actions, rewards, q_values, policy):
    return discounted(
        discount, ts1, ts2,
        R.expected_0(states, q_values, policy),
        R.expected(states, actions, rewards, q_values, policy))

def tree_discounted(discount, ts1, ts2, states, actions, rewards, q_values, policy):
    return discounted_terminal(
        discount, ts1, ts2,
        R.expected(states, actions, rewards, q_values, policy))

# def sarsa(env, discount, train_ts, epsilon, lr, nstep):
#     # Linear Model
#     weights = Linear.weights(env.n_features, env.n_actions)
#     # Functions
#     def action_do(t0, ts, states, actions, rewards):
#         Linear.target_update_q_value(
#             weights, lr, states[t0], actions[t0],
#             G.sarsa(
#                 discount, t0, ts, states, actions, rewards,
#                 Linear.q_value(weights)))
#     def end(t0, ts, states, actions, rewards):
#         Linear.target_update_q_value(
#             weights, lr, states[t0], actions[t0],
#             G.sarsa_terminal(discount, t0, ts, rewards))
#     # GPI
#     GPI.nstep(
#         env, train_ts, nstep,
#         start=do_nothing,
#         act=P.e_greedy(
#             env.rand_action, epsilon,
#             Linear.q_values(weights)),
#         state_do=do_nothing,
#         action_do=action_do,
#         reward_do=do_nothing,
#         end=end)
#     return lambda epsilon: P.e_greedy_const(
#         env.rand_action, epsilon, Linear.q_values(weights))

# def qlearning(env, discount, train_ts, epsilon, lr, nstep):
#     # Linear Model
#     weights = Linear.weights(env.n_features, env.n_actions)
#     # Functions
#     def state_do(t0, ts, states, actions, rewards):
#         Linear.target_update_q_value(
#             weights, lr, states[t0], actions[t0],
#             G.qlearning(
#                 discount, t0, ts, states, actions, rewards,
#                 Linear.q_values(weights)))
#     def end(t0, ts, states, actions, rewards):
#         Linear.target_update_q_value(
#             weights, lr, states[t0], actions[t0],
#             G.qlearning_terminal(
#                 discount, t0, ts, states, actions, rewards,
#                 Linear.q_values(weights)))
#     GPI.nstep(
#         env, train_ts, nstep,
#         start=do_nothing,
#         act=P.e_greedy(
#             env.rand_action, epsilon,
#             Linear.q_values(weights)),
#         state_do=state_do,
#         action_do=do_nothing,
#         reward_do=do_nothing,
#         end=end)

def sole_iter(env, discount, iterations=10):
    # Define State Action Space
    n_states = env.n_states + 1 # Plus 1 for Absorbing State
    n_actions = env.n_actions
    n_state_actions = n_states * n_actions
    # Map State-Action
    msa = lambda state, action: state * n_actions + action
    # Build Reward & Transition Matrix
    R = np.zeros(n_state_actions)
    P = np.zeros((n_state_actions, n_states))
    for state in range(n_states - 1):
        for action in range(n_actions):
            for p, s, r, d in env.P[state][action]:
                R[msa(state, action)] += p * r
                if d: s = n_states - 1 # To Absorbing State
                P[msa(state, action), s] += p
    # Handle Absorbing State
    for action in range(n_actions):
        state = n_states - 1
        P[msa(state, action), state] = 1
    # GPI
    I = np.identity(n_state_actions)
    Q = np.zeros(n_state_actions)
    PI = np.zeros((n_states, n_state_actions))
    # Solve for Q
    def solve_Q():
        nonlocal Q
        Q = np.linalg.solve(I - discount * P @ PI, R)
        # new_Q = np.linalg.solve(I - discount * P @ PI, R)
        # np.copyto(Q, new_Q)
    # Build Policy Matrix
    def build_PI():
        PI.fill(0)
        for state in range(n_states):
            state_action_0 = msa(state, 0)
            state_action_n = msa(state, n_actions)
            sub_Q = Q[state_action_0:state_action_n]
            max_value = np.amax(sub_Q)
            indices = np.argwhere(sub_Q == max_value)
            indices += state_action_0
            PI[state][indices] = 1 / len(indices)
            # greedy_state_action = \
            #     state_action_0 + np.argmax(
            #         Q[state_action_0:state_action_n])
            # PI[state, greedy_state_action] = 1
    # Step
    def step():
        build_PI()
        solve_Q()
    # Q Function
    qf = lambda state, action=None: (
        Q[msa(state, action)] if action
        else Q[msa(state, 0):msa(state, n_actions)])
    return qf, step