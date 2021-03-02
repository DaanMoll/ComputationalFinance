import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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

    return stock_prices


def analytical_asian_option_valuation():
    sigma_tilde = sigma * np.sqrt((2 * n + 1) / (6 * n + 6))
    r_tilde = 0.5 * (r - 0.5 * sigma * sigma + sigma_tilde * sigma_tilde)

    d1 = (np.log(stock_price / strike_price) + t * (r_tilde + 0.5 * sigma_tilde * sigma_tilde)) / (
                np.sqrt(t) * sigma_tilde)
    d2 = d1 - np.sqrt(t) * sigma_tilde

    return np.exp(-r * t) * (stock_price * np.exp(r_tilde * t) * norm.cdf(d1) - strike_price * norm.cdf(d2))


n_paths = int(1e6)
all_values = 0

for _ in tqdm(range(n_paths)):
    payoff = np.maximum(sum(euler_option_valuation()) / n - strike_price, 0)
    option_value = np.exp(-r * t) * payoff
    all_values += option_value

average_value = all_values / n_paths


print(f'Analytical Asian option price value: {analytical_asian_option_valuation():.2f}')
print(f'Euler Asian option price value: {analytical_asian_option_valuation():.2f}')
