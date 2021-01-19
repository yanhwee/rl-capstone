from v1.agents import *
from v1.interact import Interact
from v1.preprocess import *
from v1.policies import *
from v1.utils import (
    simple_bar, simple_line, simple_scatter, 
    mae, argmax_equal_percent)

class Sarsa:
    def __init__(self, env, gamma, epsilon_start, epsilon_end, alpha_start, alpha_end, max_eps, n_step, eval_q_table, init_q_table=None, tqdm_nested=False):
        # Define Constants
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        epsilon = epsilon_start
        alpha = alpha_start
        epsilon_decay = (epsilon_start - epsilon_end) / max_eps
        alpha_decay = (alpha_start - alpha_end) / max_eps
        # Sarsa
        agent = SarsaAgent(
            n_actions, n_states, gamma, 
            EGreedy(epsilon), n_step, alpha)
        if init_q_table is not None:
            agent.weights = init_q_table.T
        # Collect Stats
        eps_q_loss = [0] * max_eps
        eps_pi_opt = [0] * max_eps
        def eps_end(i):
            q_table = agent.weights.T
            eps_q_loss[i] = mae(q_table, eval_q_table)
            eps_pi_opt[i] = argmax_equal_percent(q_table, eval_q_table)
            agent.target_policy.epsilon -= epsilon_decay
            agent.learning_rate -= alpha_decay
        # Run
        eps_act, eps_obs, eps_rewards, eps_states, eps_actions = \
            Interact.train(env, agent, max_eps, eps_end=eps_end, tqdm_nested=tqdm_nested)
        # Collect Stats
        eps_acc_rewards = [sum(ep_rewards) for ep_rewards in eps_rewards]
        # Save
        self.env, self.gamma, self.epsilon_start, self.epsilon_end, self.alpha_start, self.alpha_end, self.max_eps, self.n_step, self.n_states, self.n_actions, self.epsilon, self.alpha, self.epsilon_decay, self.alpha_decay, self.agent, self.eps_q_loss, self.eps_pi_opt, self.eps_acc_rewards, self.eps_act, self.eps_obs, self.eps_rewards, self.eps_states, self.eps_actions = env, gamma, epsilon_start, epsilon_end, alpha_start, alpha_end, max_eps, n_step, n_states, n_actions, epsilon, alpha, epsilon_decay, alpha_decay, agent, eps_q_loss, eps_pi_opt, eps_acc_rewards, eps_act, eps_obs, eps_rewards, eps_states, eps_actions
    def test(self, delay, limit=None):
        env, agent = self.env, self.agent
        Interact.test(env, agent, delay=delay, limit=limit)
    def stats(self, title=None):
        env, gamma, epsilon_start, epsilon_end, alpha_start, alpha_end, max_eps, n_step, n_states, n_actions, epsilon, alpha, epsilon_decay, alpha_decay, agent, eps_q_loss, eps_pi_opt, eps_acc_rewards, eps_act, eps_obs, eps_rewards, eps_states, eps_actions = self.env, self.gamma, self.epsilon_start, self.epsilon_end, self.alpha_start, self.alpha_end, self.max_eps, self.n_step, self.n_states, self.n_actions, self.epsilon, self.alpha, self.epsilon_decay, self.alpha_decay, self.agent, self.eps_q_loss, self.eps_pi_opt, self.eps_acc_rewards, self.eps_act, self.eps_obs, self.eps_rewards, self.eps_states, self.eps_actions
        simple_scatter(eps_acc_rewards, 1, 'Total Rewards', 'Episodes', title)
        simple_line(eps_acc_rewards, 1, 'Total Rewards', 'Episodes', title)
        simple_line(eps_q_loss, 1, 'Average State-Action Loss', 'Episodes', title)
        simple_line(eps_pi_opt, 1, '% Optimal Policy', 'Episodes', title)
        print('Final Q Values MAE', eps_q_loss[-1])
        print('Final % Optimal Policy', eps_pi_opt[-1])

# class Sarsa:
#     def __init__(self, env, gamma, epsilon_start, epsilon_end, alpha_start, alpha_end, max_eps, n_step, eval_q_table):
#         # Define Constants
#         n_states = env.observation_space.n
#         n_actions = env.action_space.n
#         max_timestep = env._max_episode_steps
#         max_length = max_timestep + 1
#         epsilon = epsilon_start
#         alpha = alpha_start
#         epsilon_decay = (epsilon_start - epsilon_end) / max_eps
#         alpha_decay = (alpha_start - alpha_end) / max_eps
#         # Sarsa
#         q_table = np.zeros((n_states, n_actions))
#         eps_acc_rewards = [0] * max_eps
#         eps_q_loss = [0] * max_eps
#         eps_pi_opt = [0] * max_eps
#         rewards = [0] * max_length
#         states = [0] * max_length
#         actions = [0] * max_length
#         for i in tqdm(range(max_eps)):
#             state = env.reset()
#             reward = 0
#             acc_reward = 0
#             for t in range(min(max_timestep)):
#                 rewards[t] = reward
#                 states[t] = state
#                 action = (
#                     np.random.randint(n_actions)
#                     if np.random.random() < epsilon
#                     else np.argmax(q_table[state]))
#                 actions[t] = action
#                 state, reward, done, info = env.step(action)
#                 acc_reward += reward
#                 if done: break
#             t += 1
#             rewards[t] = reward
#             states[t] = state
#             actions[t] = 0