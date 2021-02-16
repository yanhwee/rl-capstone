from v2.utils import argmax

def q_mae(states, n_actions, qf1):
    return lambda qf2: \
        sum(sum(abs(qf1(state) - qf2(state))) for state in states) / len(states) / n_actions

def policy_diff(states, qf1, qf2):
    argmax(qf1(state)) == argmax(qf2(state))