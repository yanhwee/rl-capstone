# rl-capstone
### Run
1. `setup.bat`
2. `jupyter lab`

### Troubleshooting
1. TQDM Import Error
    - `conda install nodejs`
    - `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
    - https://stackoverflow.com/questions/53247985/tqdm-4-28-1-in-jupyter-notebook-intprogress-not-found-please-update-jupyter-an
    - https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension
2. matplotlib interactive plots
    - Follow step 1 first
    - `conda install ipympl`
    - `jupyter labextension install jupyter-matplotlib`
    - https://stackoverflow.com/questions/50149562/jupyterlab-interactive-plot

### Example & Experiments
My apologies 😅 the workspace is kinda messy.
Experiment | Algorithm | Path
--- | --- | ---
Grid World 4 x 4 | Exact Prediction | `gridworld/ex_3_8.py`
Grid World 5 x 5 | Value Iteration | `gridworld/ex_4_1.py`
FrozenLake & Taxi-v3 | Exact Prediction, Value Iteration | `sole_vs_dp/`
Taxi-v3 | Exact Prediction | `sole/`
Taxi-v3 | Value Iteration | `dp/`
Taxi-v3 | Monte Carlo | `montecarlo/`
Taxi-v3 | SARSA | `sarsa/`
Taxi-v3 | Q-Learning | `qlearning/`
Taxi-v3 | Expected SARSA | `expectedsarsa/`
Cliff Walking | SARSA, Q-Learning, Expected SARSA | `gridworld/ex_6_6.py`
Random Walk | n-step SARSA | `gridworld/random_walk.py`
Mountain Car | Tabular Q-Learning | `experiments/mountaincar.ipynb`
Mountain Car | Naive Linear Q-Learning | `experiments/mountaincar-flat.ipynb`
Mountain Car | Fourier Linear Q-Learning | `experiments/mountaincar-fourier.ipynb`
Mountain Car | DQN | `dqn/mountaincar2.ipynb`