import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import errors, njit
import pickle 
t = 1
dt = 1 / 365
n = int(t / dt)
r = 0.06
sigma = 0.2
strike_price = 99
stock_price = 100
epsilon = 0.1 #0.1 0.001
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

@njit
def get_delta(same_seed=False, capped_value=0):
    values = []
    perturbed_values = []

    for _ in range(n_trajectories):
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

    average_value = np.sum(np.array(values)) / n_trajectories
    perturbed_average_value = np.sum(np.array(perturbed_values)) / n_trajectories

    return (perturbed_average_value - average_value) / epsilon


deltas = []
errors = []
for i in tqdm(range(100)):
    delta = get_delta()
    deltas.append(delta)
    error=(0.673735-delta)/0.673735
    errors.append(error)


deltas_s=[]
errors_s=[]
for i in tqdm(range(100)):
    delta = get_delta(same_seed=True)
    deltas_s.append(delta)
    error= (0.673735-delta)/0.673735
    errors_s.append(error)

value_cap = 1
print(f'delta (different seed, epsilon={epsilon}) = {np.mean(deltas), np.std(deltas), np.mean(errors)}')
print(f'delta (same seed, epsilon={epsilon}) = {np.mean(deltas_s), np.std(deltas_s), np.mean(errors_s)}')
print(rf'delta (different seed, value capped at {value_cap}) = {get_delta(capped_value=value_cap):.2f}')
print(rf'delta (same seed, value capped at {value_cap}) = {get_delta(same_seed=True, capped_value=value_cap):.2f}')


# Digital option
@njit
def digital_delta_bumped(n_iter, epsilon):
    seed = np.random.seed(np.random.randint(300))
    bumped_values, final_st = euler_digital_valuation(
        n_iter, epsilon=epsilon)
    unbumped_values, final_st_2 = euler_digital_valuation(
        n_iter, epsilon=0)
    delta = (np.mean(np.array(bumped_values)) -
             np.mean(np.array(unbumped_values))) / epsilon
    mean_st_true = (final_st+final_st_2)/2
    analytical = get_analytical_delta(np.mean(np.array(mean_st_true)))
    error = abs(delta-analytical)
    return delta, error


@njit
def euler_digital_valuation(n_iter, epsilon):
    ''' Returns the values of n_iter digital options'''
    all_digital_valuations = []
    all_mean_st = []
    for i in range(n_iter):
        payoffs, mean_st = digital_option_bumped(epsilon)
        all_digital_valuations.append(np.mean(payoffs))
        all_mean_st.append(mean_st)

    return all_digital_valuations, np.mean(np.array(all_mean_st))


@njit
def digital_option_bumped(epsilon):
    ''' Returns the value of ONE digital option'''
    mc_payoffs = []
    all_st = []
    for _ in range(100):
        phi = np.random.standard_normal()
        ST = (S0+epsilon) * np.exp((r - 0.5 * sigma ** 2)
                                   * T + sigma * np.sqrt(T) * phi)
        all_st.append(ST)
        if ST-K > 0:
            mc_payoffs.append(1 * np.exp(-r))  # Present value of 1 euro
        else:
            mc_payoffs.append(0)  # or present value of nothing
    return np.array(mc_payoffs), np.mean(np.array(all_st))


iterations = [100, 1000, 10000, 50000, 100000]
errors = [1, 0.5, 0.01, 0.001, 0.0001]

results_2 = {}
for epsilon in errors:
    experiment_results = {}
    for itera in iterations:
        sample_delta = []
        sample_errors = []
        for samples in range(1000):
            delta, error = digital_delta_bumped(itera, epsilon)
            sample_delta.append(delta)
            sample_errors.append(error)
        experiment_results[itera] = [sample_delta, sample_errors]
    results_2[epsilon] = experiment_results
