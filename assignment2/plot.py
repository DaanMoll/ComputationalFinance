import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

sample_trajectories = 10 ** np.arange(2, 6, 0.5)

with open('option_values', 'rb') as f:
    option_values = pk.load(f)
    option_values_mean = [np.mean(i) for i in option_values]
    option_values_std = [np.std(i) for i in option_values]

plt.errorbar(sample_trajectories, option_values_mean, option_values_std, linestyle='None', marker='.', capsize=3)
plt.xscale('log')
plt.xlabel("# Sample trajectories")
plt.ylabel("Option Value")
plt.show()
