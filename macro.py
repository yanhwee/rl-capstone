import numpy as np
N_NEWLINES = 0
N_TABS = 0
text = \
'''

PI_changes, Q_changes, Q_dt
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