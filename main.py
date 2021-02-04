import numpy as np

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

    print(matrix)
    return matrix

def valueOptionMatrix(tree, T, r, K, vol, N):
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
        value = S-K
        tree[rows - 1, c] = value if value>0 else 0

    # For all other rows, we need to combine from previous rows 
    # We walk backwards, from the last row to the first row
    for i in np.arange(rows - 1)[::-1]: 
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            tree[i, j] = np.exp(-r*dt) * (p*up + (1-p)*down)

    return tree

sigma = 0.2
S = 100
T = 1
N = 50

tree = buildTree(S, sigma, T, N)

K = 99
r = 0.06
value = valueOptionMatrix(tree, T, r, K, sigma, N)

print(value)
print(value[0][0])

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