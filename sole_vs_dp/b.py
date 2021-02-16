import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    lake_sole_time = 0.06495572999999996
    lake_dp_time = 0.07271513
    taxi_sole_time = 19.91573316
    taxi_dp_time = 0.29470574000000055
    
    print(lake_sole_time / 64)
    print(lake_dp_time / 64)
    print(taxi_sole_time / 500)
    print(taxi_dp_time / 500)

    env_labels = ['FrozenLake8x8-v0 (64 states)', 'Taxi-v3 (500 states)']
    sole_times = [lake_sole_time, taxi_sole_time]
    dp_times = [lake_dp_time, taxi_dp_time]
    lake_times = [lake_sole_time, lake_dp_time]
    taxi_times = [taxi_sole_time, taxi_dp_time]
    algo_labels = ['System of Linear Equations', 'Value Iteration']

    xs = np.arange(len(env_labels))
    width = 0.35

    fig, ax = plt.subplots()
    sole_rect = ax.bar(xs - width/2, sole_times, width, label='System of Linear Equations')
    dp_rect = ax.bar(xs + width/2, dp_times, width, label='Value Iteration')

    ax.set_ylabel('Average time taken among 10 runs (sec)')
    ax.set_title('Time taken to find optimal policy')
    ax.set_xticks(xs)
    ax.set_xticklabels(env_labels)
    ax.legend()

    plt.show()

    plt.bar(algo_labels[0], lake_times[0])
    plt.bar(algo_labels[1], lake_times[1])
    plt.title('FrozenLake8x8-v0 (64 states) - Time taken to find optimal policy')
    plt.ylabel('Average time taken among 10 runs (sec)')
    plt.show()

    plt.bar(algo_labels[0], taxi_times[0])
    plt.bar(algo_labels[1], taxi_times[1])
    plt.title('Taxi-v3 (500 states) - Time taken to find optimal policy')
    plt.ylabel('Average time taken among 10 runs (sec)')
    plt.show()