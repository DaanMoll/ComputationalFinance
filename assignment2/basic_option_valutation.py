import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import pickle

t = 1
dt = 1 / 365
n = int(t / dt)
r = 0.06
sigma = 0.2
strike_price = 99
stock_price = 100

@njit
def euler_option_valuation(plot=False):

    stock_prices = np.zeros(n)
    stock_prices[0] = stock_price

    for i in range(1, n):
        phi = np.random.standard_normal()
        stock_prices[i] = stock_prices[i-1] * (1 + (r * dt + sigma * phi * np.sqrt(dt)))

    # if plot:
    #     plt.plot(range(n), stock_prices, 'r-')
    #     plt.show()

    return stock_prices[-1]


# option_values = []
# std = []
# sample_trajectories = 10 ** np.arange(2, 6, 0.25)

# for n_trajectories in sample_trajectories:
#     n_trajectories = int(n_trajectories)
#     all_values = []

#     for _ in tqdm(range(n_trajectories)):
#         payoff = np.maximum(euler_option_valuation() - strike_price, 0)
#         option_value = np.exp(-r * t) * payoff
#         all_values.append(option_value)

#     # average_value = all_values / n_trajectories
#     option_values.append(np.mean(all_values))
#     std.append(np.std(all_values))


# plt.plot(sample_trajectories, option_values, 'r-')
# plt.xlim(min(sample_trajectories), max(sample_trajectories))
# plt.xscale('log')
# plt.xlabel("# Sample trajectories")
# plt.ylabel("Option Value")
# plt.show()


# Perform numerical tests for varying values for the strike and the volatility parameter.


normal_results = {}

variance_results = {}
for var in range(1):
    traj_results  = {}
    sample_trajectories = 10 ** np.arange(2, 5, 0.5)
    for n_trajectories in tqdm(sample_trajectories):
        n_trajectories = int(n_trajectories)
        all_values = []
        # One repetition is one option value. N_trajectories -> n_option values
        for _ in range(n_trajectories):
            values = []
            for _ in range(1000):
                payoff = np.maximum(euler_option_valuation() - strike_price, 0)
                option_value = np.exp(-r * t) * payoff
                values.append(option_value)
            average_value = np.mean(values)
            all_values.append(average_value)

        n_trajectory_result = [np.mean(all_values), np.std(all_values)]
        traj_results[n_trajectories]=n_trajectory_result
    normal_results = traj_results

with open('normal.pickle', 'wb') as handle:
    pickle.dump(normal_results, handle, protocol=pickle.HIGHEST_PROTOCOL)



variances = [0.10, 0.20, 0.40, 0.60]
strikes = [99, 90, 80, 70, 60, 50]


strike_results = {}
for strike in strikes:
    strike_price = strike
    sigma = var
    traj_results  = {}
    sample_trajectories = 10 ** np.arange(2, 5, 0.5)
    for n_trajectories in tqdm(sample_trajectories):
        n_trajectories = int(n_trajectories)
        all_values = []
        # One repetition is one option value. N_trajectories -> n_option values
        for _ in range(n_trajectories):
            values = []
            for _ in range(1000):
                payoff = np.maximum(euler_option_valuation() - strike_price, 0)
                option_value = np.exp(-r * t) * payoff
                values.append(option_value)
            average_value = np.mean(values)
            all_values.append(average_value)

        n_trajectory_result = [np.mean(all_values), np.std(all_values)]
        traj_results[n_trajectories]=n_trajectory_result
    strike_results[strike] = traj_results

with open('strike.pickle', 'wb') as handle:
    pickle.dump(strike_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

variances = [0.10, 0.20, 0.40, 0.60]

t = 1
dt = 1 / 365
n = int(t / dt)
r = 0.06
sigma = 0.2
strike_price = 99
stock_price = 100


variance_results = {}
for var in variances:
    sigma = var
    traj_results  = {}
    sample_trajectories = 10 ** np.arange(2, 5, 0.5)
    for n_trajectories in tqdm(sample_trajectories):
        n_trajectories = int(n_trajectories)
        all_values = []
        # One repetition is one option value. N_trajectories -> n_option values
        for _ in range(n_trajectories):
            values = []
            for _ in range(1000):
                payoff = np.maximum(euler_option_valuation() - strike_price, 0)
                option_value = np.exp(-r * t) * payoff
                values.append(option_value)
            average_value = np.mean(values)
            all_values.append(average_value)

        n_trajectory_result = [np.mean(all_values), np.std(all_values)]
        traj_results[n_trajectories]=n_trajectory_result
    strike_results[var] = traj_results

with open('variances.pickle', 'wb') as handle:
    pickle.dump(variance_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
