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
epsilon = 0.01
n_trajectories = int(1e5)


@njit
def euler_option_valuation(same_seed=False):
    stock_prices = np.zeros(n)
    stock_prices[0] = stock_price

    perturbed_stock_prices = np.zeros(n)
    perturbed_stock_prices[0] = stock_price + epsilon

    for i in range(1, n):
        phi = np.random.standard_normal()
        stock_prices[i] = stock_prices[i - 1] * (1 + (r * dt + sigma * phi * np.sqrt(dt)))
        if not same_seed:
            phi = np.random.standard_normal()
        perturbed_stock_prices[i] = perturbed_stock_prices[i - 1] * (1 + (r * dt + sigma * phi * np.sqrt(dt)))

    return stock_prices[-1], perturbed_stock_prices[-1]


def get_delta(same_seed=False, capped_value=0):
    values = []
    perturbed_values = []

    for _ in tqdm(range(n_trajectories)):
        price, perturbed_price = euler_option_valuation(same_seed)

        payoff = np.maximum(price - strike_price, 0)
        payoff *= np.exp(-r * t)
        if capped_value > 0:
            values.append(payoff if payoff < capped_value else capped_value)
        else:
            values.append(payoff)

        perturbed_payoff = np.maximum(perturbed_price - strike_price, 0)
        perturbed_payoff *= np.exp(-r * t)
        if capped_value > 0:
            perturbed_values.append(perturbed_payoff if perturbed_payoff < 1 else 1)
        else:
            perturbed_values.append(perturbed_payoff)

    average_value = sum(values) / n_trajectories
    perturbed_average_value = sum(perturbed_values) / n_trajectories

    return (perturbed_average_value - average_value) / epsilon


value_cap = 1
print(rf'delta (different seed) = {get_delta():.2f}')
print(rf'delta (same seed) = {get_delta(same_seed=True):.2f}')
print(rf'delta (different seed, value capped at {value_cap}) = {get_delta(capped_value=value_cap):.2f}')
print(rf'delta (same seed, value capped at {value_cap}) = {get_delta(same_seed=True, capped_value=value_cap):.2f}')
