import numpy as np
import math
from scipy.stats import norm
import random
import matplotlib.pyplot as plt

def buildTree(S, vol, T, N): 
    dt = T / N
    matrix = np.zeros((N + 1, N + 1))

    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    matrix[0, 0] = S

    # Iterate over the lower triangle
    for i in np.arange(1, N + 1): # iterate over rows
        for j in np.arange(i + 1): # iterate over columns
            # Hint: express each cell as a combination of up and down moves 
            if j == 0:
                matrix[i, j] = matrix[i-1, j] * d
            else:
                matrix[i, j] = matrix[i-1, j-1] * u

    return matrix

def valueOptionMatrix(tree, T, r, K, vol, N, option, origin):
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))
    p = (np.exp(r*dt) - d) / (u - d)

    columns = tree.shape[1] 
    rows = tree.shape[0]
    # Walk backward , we start in last row of the matrix
    # Add the payoff function in the last row 
    for c in np.arange(columns):
        S = tree[rows - 1, c] # value in the matrix 
        if option == "Call":
            tree[rows - 1, c] = max(0, S-K)
        else:
            tree[rows - 1, c] = max(0, K-S)

    # For all other rows, we need to combine from previous rows 
    # We walk backwards, from the last row to the first row
    for i in np.arange(rows - 1)[::-1]: 
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            if origin == "American":
                tree[i, j] = max(tree[i, j] - K, np.exp(-r*dt) * (p*up + (1-p)*down))
            else:
                tree[i, j] = np.exp(-r*dt) * (p*up + (1-p)*down)
    return tree

sigma = 0.2
S = 100
T = 1
N = 50
tree = buildTree(S, sigma, T, N)

K = 99
r = 0.06
option = "Call"
origin = "European"
tree = valueOptionMatrix(tree, T, r, K, sigma, N, option, origin)
print(f"{origin} {option} option value =", tree[0][0])

# blackscholes
t = 0
tau = T - t
d1 = np.log(S/K) + (r + 0.5 * sigma**2) * tau
d1 = d1 / (sigma * np.sqrt(tau))
d2 = d1 - sigma * np.sqrt(tau)
Call = S * norm.cdf(d1) - np.exp(-r*tau) * K * norm.cdf(d2)
Put = (np.exp(-r*tau) * K * norm.cdf(-d2)) - (S * norm.cdf(-d1))

if option == "Call":
    print("Black scholes call value =", Call)
else:
    print("Black shcoles put value =", Put)

# 1.5
dt = T / N
u = np.exp(sigma * np.sqrt(dt))
d = np.exp(-sigma * np.sqrt(dt))

fu = tree[1][1]
fd = tree[1][0]

delta = (fu - fd) / (S * u - S * d)
print("delta =", delta)
print("analytical value =", norm.cdf(d1))

origin = "American"
tree = buildTree(S, sigma, T, N)
tree = valueOptionMatrix(tree, T, r, K, sigma, N, option, origin)
print(f"{origin} {option} option value = ", tree[0][0])

# example from appendix A.1
# u = 2
# d = 0.5
# s0 = 4
# ert = 1.25
# K = 5
# fu = 8 - K
# fd = 0

# emin = np.exp(-0.22314355131)
# print(emin)

# p = (ert - d) / (u - d)
# f0 = emin * (p*fu + (1-p)*fd)
# print(f0)

# 2.2
sigma = 0.2
r = 0.06
S0 = 100
K = 99
T = 1
M = 365 # weeks?
dt = T/M

d1s = []
all_S = [S0]
tau = 1
for m in range(M):
    z = np.random.normal(0, 1)
    current_S = all_S[-1]
    S = current_S + r*current_S*dt + sigma * current_S * np.sqrt(dt) * z

    d1 = np.log(S/K) + (r + 0.5 * sigma**2) * tau
    d1 = d1 / (sigma * np.sqrt(tau))

    d1s.append(d1)
    all_S.append(S)
    tau -= dt


plt.plot(range(M + 1), all_S)
plt.xlabel("Time steps")
plt.ylabel("Stock price")
# plt.show()
