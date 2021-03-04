import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

# Euler, bump and revalue
t = 1
M = 365
dt = 1 / M
n = int(t / dt)
r = 0.06
sigma = 0.2
strike_price = 99
stock_price = 100
epsilon = 0.01

@njit
def euler_option_valuation(epsilon, plot=False):
    stock_prices = np.zeros(n)
    stock_prices_eps = np.zeros(n)

    stock_prices[0] = stock_price
    stock_prices_eps[0] = stock_price + epsilon

    for i in range(1, n):
        phi = np.random.standard_normal()
        stock_prices[i] = stock_prices[i-1] * (1 + (r * dt + sigma * phi * np.sqrt(dt)))
        stock_prices_eps[i] = stock_prices_eps[i-1] * (1 + (r * dt + sigma * phi * np.sqrt(dt)))

    return stock_prices[-1], stock_prices_eps[-1]


def delta_valuation():
    all_values = 0
    eps_values = 0

    runs = 100000
    for _ in tqdm(range(runs)):
        euler = euler_option_valuation(epsilon=epsilon)

        payoff = np.maximum(euler[0] - strike_price, 0)
        option_value = np.exp(-r * t) * payoff
        if option_value > 1:
            option_value = 1
        all_values += option_value

        payoff_eps = np.maximum(euler[1] - strike_price, 0)
        option_value_eps = np.exp(-r * t) * payoff_eps
        if option_value_eps > 1:
            option_value_eps = 1
        eps_values += option_value_eps

    average_value = all_values / runs
    eps_value = eps_values / runs
    
    return average_value, eps_value


both = delta_valuation()

Vs = both[0]
print("Vs", Vs)

Vseps = both[1]
print("V eps:", Vseps)

delta = (Vseps - Vs) / epsilon
print(delta)
