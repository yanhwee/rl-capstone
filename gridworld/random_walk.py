from v2.env import DiscreteEnv, PreprocessLinearEnv
from v2.sole import sole
from v2.features import tab2lin
from v2.agents import sarsa
from v2.utils import linear_decay_clip
from random import randrange, choices
from math import sqrt
import numpy as np
from pathos.multiprocessing import ProcessingPool
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class RandomWalk(DiscreteEnv):
    def __init__(self, n_states):
        n_states = n_states + 1
        n_actions = 1
        P = {
            state: { action: None for action in range(n_actions) }
            for state in range(n_states) }
        for state in range(n_states - 1):
            P[state][0] = [
                (0.5, state - 1, 0, False),
                (0.5, state + 1, 0, False)]
        end_state = n_states - 1
        P[0][0][0] = (0.5, end_state, 0, True)
        P[n_states - 2][0][1] = (0.5, end_state, 1, True)
        P[end_state][0] = [(1, end_state, 0, True)]
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.state = 0
    def rand_state(self):
        return randrange(self.n_states)
    def rand_action(self):
        return randrange(self.n_actions)
    def reset(self):
        self.state = self.n_states // 2 - 1
        return self.state
    def step(self, action):
        scenarios = self.P[self.state][action]
        weights = [p for p, s, r, d in scenarios]
        p, s, r, d = choices(scenarios, weights)[0]
        self.state = s
        return s, r, d

if __name__ == '__main__':
    n_states = 19
    env = RandomWalk(n_states)
    tl = tab2lin(env.n_states)
    env = PreprocessLinearEnv(env, tl)
    opt_v = (np.arange(n_states) + 1) / (n_states + 1)
    qf2v = lambda qf: [qf(tl(state), 0) for state in range(n_states)]
    mae = lambda v1, v2: sum(abs(v2 - v1)) / len(v1)
    rmse = lambda v1, v2: sqrt(sum((v2 - v1) ** 2) / len(v1))
    qf_mae = lambda qf: mae(opt_v, qf2v(qf))
    qf_rmse = lambda qf: rmse(opt_v, qf2v(qf))
    qfs_mae = lambda qfs: sum(qf_mae(qf) for qf in qfs) / len(qfs)
    qfs_rmse = lambda qfs: sum(qf_rmse(qf) for qf in qfs) / len(qfs)
    tasks_get = lambda tasks: \
        [task.get() for task in tasks]
    algo_tasks_get = lambda tasks: \
        zip(*tasks_get(tasks))
    with ProcessingPool() as p:
        N = 500
        run_sarsa = lambda lr, nstep: \
            p.apipe(
                sarsa, env.copy(), discount=1, train_ts=int(1e3),
                epsilon=linear_decay_clip(1, 1, 1),
                lr=lr, nstep=nstep)
        run_sarsas = lambda n, lr, nstep: \
            [run_sarsa(lr, nstep) for _ in range(n)]
        lrs = np.geomspace(1, 2, 30) - 1
        # lrs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        nsteps = [1, 2, 4, 8, 16, 32, 64]
        nstep_lr_tasks = []
        for nstep in nsteps:
            lr_tasks = []
            for lr in lrs:
                lr_tasks.append(run_sarsas(N, lr, nstep))
            nstep_lr_tasks.append(lr_tasks)
        nstep_lr_qfs = []
        for nstep_i in tqdm(range(len(nsteps))):
            lr_qfs = []
            for lr_i in range(len(lrs)):
                hists, qfs = algo_tasks_get(
                    nstep_lr_tasks[nstep_i][lr_i])
                lr_qfs.append(qfs)
            nstep_lr_qfs.append(lr_qfs)
        nstep_lr_maes = []
        for nstep_i in range(len(nsteps)):
            lr_maes = []
            for lr_i in range(len(lrs)):
                qfs = nstep_lr_qfs[nstep_i][lr_i]
                lr_maes.append(qfs_mae(qfs))
            nstep_lr_maes.append(lr_maes)
        nstep_lr_rmses = []
        for nstep_i in range(len(nsteps)):
            lr_rmses = []
            for lr_i in range(len(lrs)):
                qfs = nstep_lr_qfs[nstep_i][lr_i]
                lr_rmses.append(qfs_rmse(qfs))
            nstep_lr_rmses.append(lr_rmses)
    for i, nstep in enumerate(nsteps):
        plt.plot(lrs, nstep_lr_maes[i], label=nstep)
    plt.legend()
    plt.title('MAE')
    plt.title('Random Walk (19 states) (1000 steps)')
    plt.ylabel('Average MAE error among all states')
    plt.xlabel('Learning Rate (alpha)')
    plt.show()
    
    for i, nstep in enumerate(nsteps):
        plt.plot(lrs, nstep_lr_rmses[i], label=nstep)
    plt.legend()
    plt.title('Random Walk (19 states) (1000 steps)')
    plt.ylabel('Average RMS error among all states')
    plt.xlabel('Learning Rate (alpha)')
    plt.show()

    # train_ts = int(1e2)
    # rets = []
    # for i in tqdm(range(10)):
    #     hist, qf = sarsa(
    #         env, discount=1, train_ts=train_ts,
    #         epsilon=linear_decay_clip(1, 1, 1),
    #         lr=0.4, nstep=64)
    #     n_eps = len(hist['eps_rewards'])
    #     rets.append(train_ts / n_eps)
    # print(sum(rets) / len(rets))
    # print(qf2v(qf))
    # print(qf_mae(qf))
    

    # qf = sole(env, 1 - 1e-9)
    # for state in range(n_states):
    #     print(qf(state, 0) * (n_states + 1))