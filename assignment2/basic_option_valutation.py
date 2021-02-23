import numpy as np
import matplotlib.pyplot as plt

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


euler_option_valuation(plot=True)
