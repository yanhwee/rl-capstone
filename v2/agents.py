import v2.returns as G
import v2.gpi as GPI
import v2.policies as P
import v2.linear as Linear
from v2.utils import do_nothing

def nstep_linear(
    env, train_ts, epsilon, lr, nstep,
    need_action, g, g_0):
    # Linear Model
    weights = Linear.weights(env.n_features, env.n_actions)
    # Functions
    def update_weights(g):
        return lambda t0, ts, states, actions, rewards: \
            Linear.update_q_target(
                weights, lr, states[t0], actions[t0], g(
                    t0, ts, states, actions, rewards,
                    Linear.qf(weights), P.e_greedy(
                        env.n_actions, epsilon, Linear.qf(weights))))
    # GPI
    history = GPI.nstep(
        env, train_ts, nstep,
        start=do_nothing,
        act=P.act_e_greedy(
            env.rand_action, epsilon,
            Linear.qf(weights)),
        state_do=(
            do_nothing if need_action else update_weights(g)),
        action_do=(
            update_weights(g) if need_action else do_nothing),
        reward_do=do_nothing,
        end=update_weights(g_0))
    # History & Q Function
    return history, Linear.qf(weights)

def sarsa(env, discount, train_ts, epsilon, lr, nstep):
    return nstep_linear(
        env, train_ts, epsilon, lr, nstep, need_action=True,
        g=lambda t0, ts, states, actions, rewards, qf, policy: \
            G.sarsa(discount, t0, ts, states, actions, rewards, qf),
        g_0=lambda t0, ts, states, actions, rewards, qf, policy: \
            G.sarsa_terminal(discount, t0, ts, rewards))

def montecarlo(env, discount, train_ts, epsilon, lr):
    return sarsa(env, discount, train_ts, epsilon, lr, int(1e6))

def qlearning(env, discount, train_ts, epsilon, lr, nstep):
    return nstep_linear(
        env, train_ts, epsilon, lr, nstep, need_action=False,
        g=lambda t0, ts, states, actions, rewards, qf, policy: \
            G.qlearning(discount, t0, ts, states, actions, rewards, qf),
        g_0=lambda t0, ts, states, actions, rewards, qf, policy: \
            G.qlearning_terminal(discount, t0, ts, states, actions, rewards, qf))

def expectedsarsa(env, discount, train_ts, epsilon, lr, nstep):
    return nstep_linear(
        env, train_ts, epsilon, lr, nstep, need_action=False,
        g=lambda t0, ts, states, actions, rewards, qf, policy: \
            G.expected_sarsa(discount, t0, ts, states, rewards, qf, policy),
        g_0=lambda t0, ts, states, actions, rewards, qf, policy: \
            G.expected_sarsa_terminal(discount, t0, ts, rewards))