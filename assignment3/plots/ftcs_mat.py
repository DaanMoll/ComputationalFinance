
import numpy as np
import scipy.linalg as linalg
from tqdm import tqdm
from numba import njit

# @njit
def ftcs(N, M, T, S_0, S_max, K, r, sigma, optimal_delta=True):
    '''
    N -> Number of time steps
    M -> Number of grid spaces
    S_max -> Maximum stock price 
    '''
    S_max = 2 * S_0
    if optimal_delta:
        dt = 0.0005
        N = int(T/dt)
    else:
        # print(N  )
        dt = T / N  # Time step
    dS = S_max / M  # Space step
    # print(dt)
    #  S0   [i = 1]
    #  S    [i = 2]
    #  Smax [i = M]

    all_i = np.arange(1, M)
    all_j = np.arange(N)
    # M+1 equally spaced values of S, from 0
    all_S = np.linspace(0, S_max, M+1)

    # Populate the grid with the the values
    grid = np.zeros(shape=(M+1, N+1))
    grid[:, -1] = np.maximum(all_S - K, 0)

    # Greek Arrays
    alpha = (0.25 * dt * (sigma**2 * all_i**2 - r * all_i))*2
    beta = (-0.5 * dt * (sigma**2 * all_i**2 + r))*2
    gamma = (0.25 * dt * (sigma**2 * all_i**2 + r * all_i))*2

    # A matrix
    A = np.diag(alpha[1:], -1) + np.diag(1 + beta) + np.diag(gamma[:-1], 1)

    # Side boundary conditionS
    A[0,   0] += 2*alpha[0]
    A[0,   1] -= alpha[0]
    A[-1, -1] += 2*gamma[-1]
    A[-1, -2] -= gamma[-1]

    # Terminal Condition (call option)
    grid[:, -1] = np.maximum(all_S - K, 0)

    # Iterate over the grid
    for j in reversed(all_j):
            old_grid = grid.copy()
            grid[1:-1, j] = np.dot(A, grid[1:-1, j+1])
            grid[0, j] = 2 * grid[1, j] - grid[2, j]
            grid[-1, j] = 2 * grid[-2, j] - grid[-3, j]
            if np.isnan(grid[:, 0][int(len(grid)/2)]):
                    print("Abort")
                    # option_value = old_grid[:, 0][int(len(grid)/2)]
                    option_value = np.interp(S_0, all_S, grid[:, 0])
                    print(f"Estimated option value: {option_value}")
                    return old_grid, option_value

    # option_value = grid[:, 0][int(len(grid)/2)]
    option_value = np.interp(S_0, all_S, grid[:, 0])
    # print(f"Estimated option value: {option_value}")
    return grid, option_value


# grid, value = ftcs(1000, 100, 5/12, 50, 100, 50, 0.06, 0.4)
# grid, value = ftcs(1000, 1000, 1, 50, 200, 50, 0.04, 0.3)
