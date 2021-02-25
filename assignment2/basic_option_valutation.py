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
sample_paths = 10 ** np.arange(2, 7, 0.5)

for n_paths in sample_paths:
    n_paths = int(n_paths)
    all_values = 0

    for _ in tqdm(range(n_paths)):
        payoff = np.maximum(euler_option_valuation() - strike_price, 0)
        option_value = np.exp(-r * t) * payoff
        all_values += option_value

    average_value = all_values / n_paths
    option_values.append(average_value)

plt.plot(sample_paths, option_values, 'r-')
plt.xlim(min(sample_paths), max(sample_paths))
plt.xscale('log')
plt.xlabel("# Sample Paths")
plt.ylabel("Option Value")
plt.show()
