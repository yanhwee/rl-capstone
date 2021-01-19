import numpy as np
N_NEWLINES = 0
N_TABS = 0
text = \
'''
env, gamma, epsilon_start, epsilon_end, alpha_start, alpha_end, max_eps, n_step, n_states, n_actions, epsilon, alpha, epsilon_decay, alpha_decay, agent, eps_q_loss, eps_pi_opt, eps_acc_rewards, eps_act, eps_obs, eps_rewards, eps_states, eps_actions
'''
if __name__ == "__main__":
    # text = input()
    words = text.replace(',', ' ').split()
    words2 = [f'self.{word}' for word in words]
    wordsA = ', '.join(words)
    wordsB = ', '.join(words2)
    middle = ' = ' + ('\\\n' * N_NEWLINES) + ('\t' * N_TABS)
    print('\nSave\n')
    print(''.join([wordsB, middle, wordsA]))
    print('\nGet\n')
    print(''.join([wordsA, middle, wordsB]))