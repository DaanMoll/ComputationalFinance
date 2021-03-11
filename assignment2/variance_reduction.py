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
n_trajectories = int(1e5)


@njit
def euler_option_valuation():

    stock_prices = np.zeros(n)
    stock_prices[0] = stock_price

    for i in range(1, n):
        phi = np.random.standard_normal()
        stock_prices[i] = stock_prices[i-1] * (1 + (r * dt + sigma * phi * np.sqrt(dt)))

    return stock_prices


def analytical_asian_option_valuation(current_price=stock_price):
    sigma_tilde = sigma * np.sqrt((2 * n + 1) / (6 * n + 6))
    r_tilde = 0.5 * (r - 0.5 * sigma * sigma + sigma_tilde * sigma_tilde)

    d1 = (np.log(current_price / strike_price) + t * (r_tilde + 0.5 * sigma_tilde * sigma_tilde)) / (
                np.sqrt(t) * sigma_tilde)
    d2 = d1 - np.sqrt(t) * sigma_tilde

    return np.exp(-r * t) * (current_price * np.exp(r_tilde * t) * norm.cdf(d1) - strike_price * norm.cdf(d2))


analytical_option_value = analytical_asian_option_valuation()

def get_numerical_option_values(control_variate=False):
    values = []

    ameans = []
    gmeans = []
    analytical = analytical_asian_option_valuation()

    for _ in range(n_trajectories):
        trajectory = np.array(euler_option_valuation())
        if control_variate:
            geometric_mean = np.maximum(gmean(trajectory) - strike_price, 0)
            arithmetic_mean = np.maximum(arithmetic_mean - strike_price, 0)

            ameans.append(arithmetic_mean)
            gmeans.append(geometric_mean)
            beta = 1.034
            payoff = arithmetic_mean - beta * (geometric_mean - analytical)
        else:
            payoff = np.maximum(sum(trajectory) / n - strike_price, 0)
        payoff *= np.exp(-r * t)
        values.append(payoff)

    return values


euler_option_values_cv = get_numerical_option_values(control_variate=True)
euler_option_values = get_numerical_option_values()

euler_option_value = sum(euler_option_values) / n_trajectories
euler_option_value_cv = sum(euler_option_values_cv) / n_trajectories

print(f'Euler Asian option price value: {euler_option_value}')
print(f'Euler Asian option price variance: {np.var(euler_option_values)}')
print(f'Euler Asian option price value: {euler_option_value_cv} (Control Variate)')
print(f'Euler Asian option price variance: {np.var(euler_option_values_cv)} (Control Variate)')
print(f'Analytical Asian option price value: {analytical_option_value}')

trajectories = [10, 100, 1000, 10000, 100000]
euler_no_cv = []
euler_no_cv_err = []
euler_cv = []
euler_cv_err = []

for n_trajectories in trajectories:
    print(f"n: {n_trajectories}")
    euler_option_values_cv = get_numerical_option_values(control_variate=True)
    euler_option_value_cv = sum(euler_option_values_cv) / n_trajectories
    euler_option_values = get_numerical_option_values()
    euler_option_value = sum(euler_option_values) / n_trajectories

    euler_no_cv.append(euler_option_value)
    euler_no_cv_err.append(np.sqrt(np.var(euler_option_values))/n_trajectories)
    euler_cv.append(euler_option_value_cv)
    euler_cv_err.append(np.sqrt(np.var(euler_option_values_cv))/n_trajectories)

plt.errorbar(trajectories, euler_no_cv, euler_no_cv_err, linestyle='None', marker='.', capsize=3, label="No CV")
plt.errorbar(trajectories, euler_cv, euler_cv_err, linestyle='None', marker='.', capsize=3, label="CV")
plt.xscale('log')
plt.xlabel("# Sample trajectories")
plt.ylabel("Option Value")
plt.axhline(y=analytical_option_value, linestyle='--', color='k')
plt.legend()
plt.show()
