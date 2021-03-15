from tqdm import tqdm
import numpy as np

# N = 1000
# M = 100
# T = 5/12
# S0 = 50
# Smax = 100
# K = 50
# r = 0.06
# sigma = 0.4

# u = np.empty((N, M, M))

# dt = T/M
# dS = int((Smax)/M)

# u_top = 0 # Dirichlet Condition
# u_left = 0
# u_right = 0 # Option value at expiry
# u_bottom = 0 #BS analytical payoff

# u[:, (M-1):, :] = u_top
# u[:, :, :1] = u_left
# u[:, :1, 1:] = u_bottom
# u[:, :, (M-1):] = u_right

# for time_step in tqdm(range(0, N-1, 1)):
#     for i in range(1, M-1, dS):
#          for j in range(1, M-1, dS):
#             alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
#             beta = (1 - (dt * (sigma**2 * i**2 - r)))
#             gamma = (0.5 * dt * (sigma**2 * i**2 + r * i))
#             u[time_step+1, i, j] = alpha * u[time_step, i-1, j] + beta * u[time_step, i, j] + gamma * u[time_step, i+1, j]

# print(u[-1][int(M/2)][int(M/2)])
# u[-1]


# Version two with only a TWO DIMENSIONAL GRID


# N = 1000
# M = 100
# T = 5/12
# S0 = 50
# Smax = 100
# K = 50
# r = 0.06
# sigma = 0.4

# dt = T/M
# dS = int((Smax)/M)
# all_S = np.linspace(0, Smax, M+1)

# # Populate the grid with the the values
# u = np.zeros(shape=(M+1, N+1))
# u[:, -1] = np.maximum(all_S - K, 0)


# for time_step in tqdm(range(0, N-1, 1)):
#     u_n = u.copy()
#     for i in range(1, M-1, dS):
#          for j in range(1, M-1, dS):
#             alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
#             beta = (1 - (dt * (sigma**2 * i**2 - r)))
#             gamma = (0.5 * dt * (sigma**2 * i**2 + r * i))
#             u[i, j] = alpha * u_n[i-1, j] + \
#                 beta * u_n[ i, j] + gamma * u_n[i+1, j]

# print(u[int(M/2)][int(M/2)])
# u[]


# N = 1000
# M = 100
# T = 5/12
# S0 = 50
# Smax = 100
# K = 50
# r = 0.06
# sigma = 0.4

# u = np.empty((N, M, M))

# dt = T/M
# dS = int((Smax)/M)
# dS = 1
# all_S = np.linspace(0, Smax, M)
# # grid = np.zeros(shape=(M+1, N+1))
# # grid[:, -1] = np.maximum(all_S - K, 0)

# for time_step in tqdm(range(0, N, 1)):
#     for i in range(1, M-1, dS):
#          u[time_step, i] = np.maximum(all_S - K, 0)

# # for time_step in tqdm(range(0, N-1, 1)):
# #     for i in range(1, M-1, dS):
# #          for j in range(1, M-1, dS):
# #              u[time_step+1, i, j] = np.maximum(all_S[j] - K, 0)




# for time_step in tqdm(range(0, N-1, 1)):
#     for i in range(1, M-1, dS):
#          for j in range(1, M-1, dS):
#             alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
#             beta = (1 - (dt * (sigma**2 * i**2 - r)))
#             gamma = (0.5 * dt * (sigma**2 * i**2 + r * i))
#             u[time_step+1, i, j] = alpha * u[time_step, i-1, j] + beta * u[time_step, i, j] + gamma * u[time_step, i+1, j]

# print(u[-1][int(M/2)][int(M/2)])


N = 1000
M = 100
T = 5/12
S0 = 50
Smax = 100
K = 50
r = 0.06
sigma = 0.4

u = np.empty((N, M, M))

dt = T/M
dS = int((Smax)/M)
dS = 1
all_S = np.linspace(0, Smax, M)
# grid = np.zeros(shape=(M+1, N+1))
# grid[:, -1] = np.maximum(all_S - K, 0)

for time_step in tqdm(range(0, N, 1)):
    for i in range(1, M-1, dS):
        u[time_step, i] = np.maximum(all_S - K, 0)

# for time_step in tqdm(range(0, N-1, 1)):
#     for i in range(1, M-1, dS):
#          for j in range(1, M-1, dS):
#              u[time_step+1, i, j] = np.maximum(all_S[j] - K, 0)


for time_step in tqdm(range(0, N-1, 1)):
    for i in range(1, M-1, dS):
         for j in range(1, M-1, dS):
            alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
            beta = (1 - (dt * (sigma**2 * i**2 - r)))
            gamma = (0.5 * dt * (sigma**2 * i**2 + r * i))
            u[time_step+1, i, j] = alpha * u[time_step, i-1, j] + \
                beta * u[time_step, i, j] + gamma * u[time_step, i+1, j]

print(u[-1][int(M/2)][int(M/2)])
