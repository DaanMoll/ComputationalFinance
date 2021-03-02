import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

t = 1
dt = 1 / 365
n = int(t / dt)
r = 0.06
sigma = 0.2
strike_price = 99
stock_price = 100


def euler_option_valuation(plot=False):

    stock_prices = np.zeros(n)
    stock_prices[0] = stock_price

    for i in range(1, n):
        phi = np.random.standard_normal()
        stock_prices[i] = stock_prices[i-1] * (1 + (r * dt + sigma * phi * np.sqrt(dt)))

    if plot:
        plt.plot(range(n), stock_prices, 'r-')
        plt.show()

    return stock_prices[-1]


option_values = []
sample_trajectories = 10 ** np.arange(2, 7, 0.5)

for n_trajectories in sample_trajectories:
    n_trajectories = int(n_trajectories)
    all_values = 0

    for _ in tqdm(range(n_trajectories)):
        payoff = np.maximum(euler_option_valuation() - strike_price, 0)
        option_value = np.exp(-r * t) * payoff
        all_values += option_value

    average_value = all_values / n_trajectories
    option_values.append(average_value)

plt.plot(sample_trajectories, option_values, 'r-')
plt.xlim(min(sample_trajectories), max(sample_trajectories))
plt.xscale('log')
plt.xlabel("# Sample trajectories")
plt.ylabel("Option Value")
plt.show()
