import numpy as np
import scipy.linalg as linalg
from tqdm import tqdm
from numba import njit

# @njit
def fd_cd(N, M, T, S_0, S_max, K, r, sigma):
    '''
    N -> Number of time steps
    M -> Number of grid spaces
    S_max -> Maximum stock price 
    '''
    dt = T / N  # Time step
    S_max = 2 * S_0
    dS = S_max / M # Space step

    #  S0   [i = 1]
    #  S    [i = 2]
    #  Smax [i = M]

    all_i = np.arange(1, M)
    all_j = np.arange(N)
    all_S = np.linspace(0, S_max, M+1) # M+1 equally spaced values of S, from 0

    # Populate the grid with the the values
    grid = np.zeros(shape=(M+1, N+1))
    grid[:, -1] = np.maximum(all_S - K, 0)

    # Greek Arrays
    alpha = 0.25 * dt * (sigma**2 * all_i**2 - r * all_i)
    beta = -0.5 * dt * (sigma**2 * all_i**2 + r)
    gamma = 0.25 * dt * (sigma**2 * all_i**2 + r * all_i)

    # A and B matrices
    A = np.diag(alpha[1:], -1) + np.diag(1 + beta) + np.diag(gamma[:-1], 1)
    B = np.diag(-alpha[1:], -1) + np.diag(1 - beta) + np.diag(-gamma[:-1], 1)

    # Bottom boundary conditions (Dirichlet Condition)
    B[0,   0] -= 2*alpha[0]
    B[0,   1] += alpha[0]
    B[-1, -1] -= 2*gamma[-1]
    B[-1, -2] += gamma[-1]

    # # Side boundary conditionS
    A[0,   0] += 2*alpha[0]
    A[0,   1] -= alpha[0]
    A[-1, -1] += 2*gamma[-1]
    A[-1, -2] -= gamma[-1]

    # Terminal Condition (call option)
    grid[:, -1] = np.maximum(all_S - K, 0)

    # PLU Decomposition
    P, L, U = linalg.lu(B)
    for j in reversed(all_j):
        Ux = linalg.solve(L, np.dot(A, grid[1:-1, j+1]))
        grid[1:-1, j] = linalg.solve(U, Ux)
        grid[0, j] = 2 * grid[1, j] - grid[2, j]
        grid[-1, j] = 2 * grid[-2, j] - grid[-3, j]

    option_value = grid[:, 0][int(len(grid)/2)]
    # print(f"Estimated option value: {option_value}")
    return grid, option_value


# grid, value = fd_cd(500, 100, 1, 100, 200, 99, 0.06, 0.2)
# grid, value = fd_cd(1000, 500, 1, 50, 100, 50, 0.06, 0.4)
