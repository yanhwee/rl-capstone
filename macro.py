import numpy as np
N_NEWLINES = 0
N_TABS = 0
text = \
'''
env, gamma, epsilon_start, epsilon_end, epsilon, epsilon_decay, alpha_start, alpha_end, alpha, alpha_decay, max_eps
n_states
n_actions
max_timestep
max_length
q_table
eps_acc_rewards
eps_q_loss
eps_pi_opt
rewards
states
actions
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