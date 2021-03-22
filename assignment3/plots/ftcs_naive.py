
import numpy as np
import scipy.linalg as linalg
from tqdm import tqdm
from numba import njit

@njit
def ftcs_loop(N, M, T, S_0, S_max, K, r, sigma):
    '''
    N -> Number of time steps
    M -> Number of grid spaces
    S_max -> Maximum stock price
    '''
    # dt = T/N
    dt = 0.000001
    N = int(T/dt)
    S_max = 2 * S_0
    all_S = np.linspace(0, S_max, M+1)

    # Populate the price grid with the the values
    grid = np.zeros(M+1)
    # grid = all_S - K

    # Initial condition
    for value in range(len(grid)):
        grid[value] = max(all_S[value]-K, 0)

    # Loop
    for _ in range(N):
        grid_n = grid.copy()
        for i in range(1,M):
            alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
            beta = (1 - dt * sigma**2 * i**2 - r * dt)
            gamma = (0.5 * dt * (sigma**2 * i**2 + r * i))

            # Next grid point
            grid[i] = alpha * grid_n[i-1] + (beta) * grid_n[i] + gamma * grid_n[i+1]

    # option_value = grid[int(M/2)]
    option_value = np.interp(S_0, all_S, grid)
    # print(f"Estimated option value: {option_value}")
    return grid, option_value


grid, value = ftcs_loop(1000, 500, 5/12, 50, 100, 50, 0.06, 0.4)
value