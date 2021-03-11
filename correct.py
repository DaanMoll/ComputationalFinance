from assignment2.variance_reduction import analytical_asian_option_valuation
import math
import numpy as np
from numba import njit
from tqdm import tqdm
import pickle
# Parameters
T = 1
K = 99
S0 = 100
r = 0.06
sigma = 0.2

@njit
def digital_option(n_iter):
    deltas = []
    errors = []
    for i in range(n_iter):
        phi = np.random.standard_normal()
        ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * phi)
        true_delta = .018206369779490493
        # print(true_delta)
        # Check if digital option pays off
        if ST-K > 0:
            # Present value of 1 euro
            deltas.append(np.exp(-r * T) * 1 * (phi / (S0*sigma*np.sqrt(T))))
            errors.append(abs(true_delta-np.exp(-r * T) *
                              1 * (phi / (S0*sigma*np.sqrt(T)))))
        else:
            deltas.append(np.exp(-r * T) * 0 * (phi / (S0*sigma*np.sqrt(T))))
            errors.append(abs(true_delta-np.exp(-r * T) *
                              1 * (phi / (S0*sigma*np.sqrt(T)))))
    return deltas, errors

results = {}
iterations = [100, 1000, 5000, 10000, 50000, 100000]

for itera in tqdm(iterations):
    mean_samples = []
    mean_sample_errors =[]
    for sample_number in range(100):
        mean_deltas = []
        mean_errors = []
        for i in range(itera):
            deltas, errors = digital_option(500)
            mean_deltas.append(np.mean(np.array(deltas)))
            mean_errors.append(np.mean(np.array(errors)))
        mean_samples.append(np.mean(np.array(mean_deltas)))
        mean_sample_errors.append(np.mean(np.array(mean_errors)))
    results[itera]=[mean_samples, mean_sample_errors]

for key in results.keys():
    print(np.mean(results[key][0]))

with open('likelihood.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


exit()
# Part 2
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
    analytical = .018206369779490493
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

with open('bump.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
