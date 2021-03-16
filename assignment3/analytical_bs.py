from scipy.stats import norm
import numpy as np
def analytical_bs(T, S0, K, sigma, r):
    d1 = ((r + 0.5 * sigma**2) * T -
            np.log(K / S0)) / (sigma * np.sqrt(T))
    d2 = ((r - 0.5 * sigma**2) * T -
            np.log(K / S0)) / (sigma * np.sqrt(T))
    option_value = S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
    # print(f"Option price: {option_value}")
    return option_value

# test = analytical_bs(5/12, 50, 50, 0.4, 0.06)