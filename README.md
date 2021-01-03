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