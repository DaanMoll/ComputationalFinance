import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

t = 1
dt = 1 / 365
n = int(t / dt)
r = 0.06
sigma = 0.2
strike_price = 99
stock_price = 100


@njit
def euler_option_valuation():

    stock_prices = np.zeros(n)
    stock_prices[0] = stock_price

    for i in range(1, n):
        phi = np.random.standard_normal()
        stock_prices[i] = stock_prices[i-1] * (1 + (r * dt + sigma * phi * np.sqrt(dt)))

    return stock_prices[-1]


option_values = []
std = []
sample_trajectories = 10 ** np.arange(2, 6, 0.25)

for n_trajectories in sample_trajectories:
    n_trajectories = int(n_trajectories)
    all_values = []

    for _ in tqdm(range(n_trajectories)):
        payoff = np.maximum(euler_option_valuation() - strike_price, 0)
        option_value = np.exp(-r * t) * payoff
        all_values.append(option_value)

    # average_value = all_values / n_trajectories
    option_values.append(np.mean(all_values))
    std.append(np.std(all_values))


plt.plot(sample_trajectories, option_values, 'r-')
plt.xlim(min(sample_trajectories), max(sample_trajectories))
plt.xscale('log')
plt.xlabel("# Sample trajectories")
plt.ylabel("Option Value")
plt.show()


n_trajectories = int(1e5)

for sigma in [0.2, 0.4, 0.6, 0.8, 1.0]:
    all_values = 0
    payoffs = []
    for _ in range(n_trajectories):
        payoff = np.maximum(strike_price - euler_option_valuation(), 0)
        option_value = np.exp(-r * t) * payoff
        payoffs.append(payoff)
        all_values += option_value
    average_value = all_values / n_trajectories
    print(f'K: {strike_price}, Put Value: {average_value:.2f}, SD: {np.std(payoffs)/np.sqrt(n_trajectories):.4f}')
