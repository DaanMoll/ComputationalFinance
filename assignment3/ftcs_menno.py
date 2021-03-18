import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def ftcs_menno(N, M, T, S_0, S_max, K, r, sigma):
    n = M
    r = 0.04
    sigma = 0.3
    strike_price = K
    stock_price = S_0
    max_stock_price = S_max
    t_max = T
    dt = 1 / 365
    N = t_max/dt

    def dx(nx): return nx * 0.1


    def alpha(nx): return (0.5 * sigma * sigma - r) * dt / 2 / dx(nx) / \
        dx(nx) + 0.5 * sigma * sigma * dt / dx(nx) / dx(nx)


    def beta(nx): return 1 - sigma * sigma * dt / dx(nx) / dx(nx) - r * dt


    def gamma(nx): return (r - 0.5 * sigma * sigma) * dt / 2 / dx(nx) / \
        dx(nx) + 0.5 * sigma * sigma * dt / dx(nx) / dx(nx)


    a = alpha(np.arange(1, n))
    b = beta(np.arange(1, n+1))
    c = gamma(np.arange(1, n))

    A = csr_matrix(diags([a, b, c], [-1, 0, 1]))

    v_ftcs = np.maximum(np.linspace(0, max_stock_price, n) - strike_price, 0)
    for i in range(int(N)):
        v_ftcs = spsolve(A, v_ftcs)
    return(v_ftcs)


test = ftcs_menno(100, 100, 5/12, 50, 100, 50, 0.06, 0.4)
test[int(len(test)/2)]