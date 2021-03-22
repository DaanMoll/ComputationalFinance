from assignment3.crank_nicholson import fd_cd
import matplotlib.pyplot as plt
from numba import errors
from numpy.lib.function_base import blackman
from crank_nicholson import *
from ftcs_mat import *
from analytical_bs import *
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from ftcs_naive import *

# For saving

main_name = "complex"

# Parameters
r = 0.04
sigma = 0.3
S_max = 200
K = 110
T = 1
N = 50
M = 99
midpoint = int(M/2)
S_0 = 100


# S_0 = 100 # In the money
# S_0 = 110 # At the money
# S_0 = 120  # Out of the money


# ----- 2D price plot ------
grid_cn, value = fd_cd(N, M, T, S_0, S_max, K, r, sigma)

all_S = np.linspace(0, S_max, M+1)
price_array_cn = []
for price in tqdm(all_S):
    grid_cn, value = fd_cd(N, M, T, price, S_max, K, r, sigma)
    price_array_cn.append(value)

grid_ftcs, value = ftcs_loop(N, M, T, price, S_max, K, r, sigma)

price_array_ftcs = []
for price in tqdm(all_S):
    grid_ftcs, value = ftcs(N, M, T, price, S_max, K, r, sigma)
    price_array_ftcs.append(value)

price_array_an = []
for price in tqdm(all_S):
    price_array_an.append(analytical_bs(T=T, S0=price, K=K, sigma=sigma, r=r))

plt.plot(all_S, price_array_an, label="BS", lw=3)
plt.plot(all_S, price_array_ftcs, label="FTCS", ls='--', lw=3)
plt.plot(all_S, price_array_cn, label="CN", lw=3, ls=":")
plt.legend()
plt.ylabel("Option price")
plt.xlabel("Stock price")
plt.tight_layout()
plt.savefig(f"{main_name}_2D_priceplot.pdf")


# ----- 3D final grid plot ------

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
# THE ACTUAL DATA SHOWS THE REVERSED AXIS BUT THE AXIS LABEL DOES NOT, COMMENT THIS OUT TO SEE TRUE GRAPH
all_S = all_S[::-1]
Y = all_S
X = np.arange(N)
# X = X.transpose()
X, Y = np.meshgrid(X, Y)
Z = grid_cn[:, :-1]
# Z = Z.transpose()

print(X.shape, Y.shape, Z.shape)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
ax.set_xlabel(xlabel="Time")
ax.set_ylabel(ylabel="Stock price")
ax.set_zlabel(zlabel="Option price")
# ax.set_title("Crank Nicholson")
# ax.invert_zaxis()

# plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(f"{main_name}_3D_cnplot.pdf")

#---------- Converge in space ------
# Show 3 different plots for 3 moneyness, repeat with more iterations and cut enough to show FTCSdevolving

#Parameters
r = 0.04
sigma = 0.3
S_max = 200
K = 110
T = 1
N = 50  # 100
results = {}
grid_sizes = np.linspace(10, 300, 150)  # try 5,500,20
moneyness = [100, 110, 120]
for money in moneyness:
    abort = True
    all_prices_cn = []
    all_prices_ftcs = []
    all_errors_cn = []
    all_errors_ftcs = []
    for grid_size in tqdm(grid_sizes):
        BSA = analytical_bs(T=T, S0=money, K=K, sigma=sigma,
                            r=r, return_delta=False)
        grid_cn, value_1 = fd_cd(N, int(grid_size), T,
                                 money, S_max, K, r, sigma)
        # grid_ftcs, value_2 = ftcs(N, int(grid_size), T, money, S_max, K, r, sigma)
        if abort:
            grid_ftcs, value_2 = ftcs(
                N, int(grid_size), T, money, S_max, K, r, sigma)
            if np.isnan(value_2):
                abort = False
                print("Nan")
        else:
            value_2 = BSA
        # Append prices
        all_prices_cn.append(value_1)
        all_prices_ftcs.append(value_2)
        all_errors_cn.append(abs(value_1-BSA)/BSA)
        all_errors_ftcs.append(abs(value_2-BSA)/BSA)
    results[money] = [all_prices_cn, all_prices_ftcs,
                      all_errors_cn, all_errors_ftcs]

# moneyness = 100
# e_1 = results[money][2]
# e_2 = results[moneyness][3]
# # plt.yscale("log")
# for value in e_2:
#     if value == 0:
#         index = e_2.index(value)
#         index -= 4
#     else:
#         index = 50

# plt.plot(list(grid_sizes), e_2, color="tab:red", ls='--')
# plt.plot(grid_sizes, e_1, label="CN")
# plt.plot(grid_sizes[0:index], e_2[0:index], label="FTCS")
# plt.ylim((0, 0.05))
# plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=((12, 6)), sharex=True, sharey=True)
# 100
e_1 = results[100][2]
e_2 = results[100][3]
# plt.yscale("log")
for value in e_2:
    if value == 0:
        index = e_2.index(value)
        index -= 4
    else:
        index = 150
index = 100
ax1.plot(list(grid_sizes)[0:index], e_2[0:index], color="tab:red", ls='--')
ax1.plot(grid_sizes, e_1, label="CN")
index = 75
ax1.plot(grid_sizes[0:index], e_2[0:index], label="FTCS")

#110

e_1 = results[110][2]
e_2 = results[110][3]
# plt.yscale("log")
for value in e_2:
    if value == 0:
        index = e_2.index(value)
        index -= 5
    else:
        index = 150
index = 100
ax2.plot(list(grid_sizes)[0:160], e_2[0:160], color="tab:red", ls='--')
ax2.plot(grid_sizes, e_1, label="CN")
index = 75
ax2.plot(grid_sizes[0:index], e_2[0:index], label="FTCS")

#120
e_1 = results[120][2]
e_2 = results[120][3]
# plt.yscale("log")
for value in e_2:
    if value == 0:
        index = e_2.index(value)
        index -= 4
    else:
        index = 150
index = 100
ax3.plot(list(grid_sizes)[0:index], e_2[0:index], color="tab:red", ls='--')
ax3.plot(grid_sizes, e_1, label="CN")
index = 75
ax3.plot(grid_sizes[0:index], e_2[0:index], label="FTCS")
for ax in (ax1, ax2, ax3):
    ax.set_xlabel('Mesh size')

ax1.set_ylabel('Relative error')
plt.tight_layout()
plt.savefig(f"{main_name}_convergance_multiplot_loop.pdf")
plt.tight_layout()
plt.ylim((0, 0.06))
plt.legend()
plt.show()


#--------- Plot delta as a factor of S ---
# Show the delta calculaion for different dS and different S
# Then show the true vs analytical
def disc(value):
    return np.exp(-r) * value


def delta(S, dS):
    grid_cn, value_1 = fd_cd(N, M, T, S, S_max, K, r, sigma)
    grid_cn, value_2 = fd_cd(N, M, T, S+dS, S_max, K, r, sigma)
    # return ((np.exp(value_1) - np.exp(value_2))/dS) *  np.exp(-r)
    return (abs(disc(value_1)-disc(value_2)))/dS


price = 100
delta = delta(price, 0.00001)
bs = analytical_bs(T=T, S0=price, K=K, sigma=sigma, r=r, return_delta=False)

results_delta = {}
for value in tqdm(np.linspace(100, 0.0001, 10)):
    all_errors = []
    for price in np.linspace(10, 250, 500):
        grid_cn, value_1 = fd_cd(N, M, T, price, S_max, K, r, sigma)
        grid_cn, value_2 = fd_cd(N, M, T, price+value, S_max, K, r, sigma)
        error = (abs(disc(value_1)-disc(value_2)))/value
        all_errors.append(error)
    results_delta[value] = all_errors

for dS in np.linspace(100, 0.0001, 10):
    if dS == 100 or dS == 0.0001 or dS == 44.4:
        plt.plot(np.linspace(10, 250, 500),
                 results_delta[dS], alpha=0.80, label=dS)
    else:
        plt.plot(np.linspace(10, 250, 500), results_delta[dS], alpha=0.80)
plt.xlabel("Stock price")
plt.ylabel("Delta value")
plt.legend()
plt.savefig(f"{main_name}_delta_s0_cn.pdf")


for dS in np.linspace(100, 0.0001, 10):
    if dS > 0.05:
        continue
    else:
        plt.plot(np.linspace(10, 250, 500), results_delta[dS], alpha=0.80)

plt.xlabel("Stock price")
plt.ylabel("Delta value")
plt.legend()
plt.tight_layout()
plt.savefig(f"{main_name}_delta_s0_cn_small.pdf")


# FTCS DELTA
results_delta = {}
for value in tqdm(np.linspace(1, 0.00001, 10)):
    all_errors = []
    for price in np.linspace(10, 250, 300):
        grid_cn, value_1 = ftcs_loop(N, M, T, price, S_max, K, r, sigma)
        grid_cn, value_2 = ftcs_loop(N, M, T, price+value, S_max, K, r, sigma)
        error = (abs(disc(value_1)-disc(value_2)))/value
        all_errors.append(error)
    results_delta[value] = all_errors


for dS in np.linspace(1, 0.00001, 10):
    if dS == 100 or dS == 0.00001 or dS == 44.4:
        plt.plot(np.linspace(10, 250, 300),
                 results_delta[dS], alpha=0.80, label=dS)
    else:
        plt.plot(np.linspace(10, 250, 300), results_delta[dS], alpha=0.80)
plt.xlabel("Stock price")
plt.ylabel("Delta value")
plt.legend()
plt.tight_layout()
plt.savefig(f"{main_name}_delta_s0_ftcs.pdf")


# FTCS DELTA error
results_delta = {}
for value in tqdm(np.linspace(1, 0.00001, 10)):
    all_errors = []
    for price in np.linspace(10, 250, 300):
        grid_cn, value_1 = fd_cd(N, M, T, price, S_max, K, r, sigma)
        grid_cn, value_2 = fd_cd(N, M, T, price+value, S_max, K, r, sigma)
        error = (abs(disc(value_1)-disc(value_2)))/value
        true_delta = analytical_bs(T, price, K, sigma, r, return_delta=True)
        true_error = abs(error-true_delta)
        all_errors.append(true_error)
    results_delta[value] = all_errors

for dS in np.linspace(1, 0.00001, 10):
    plt.plot(np.linspace(10, 250, 300), results_delta[dS], alpha=0.99)
plt.xlabel("Stock price")
plt.ylabel("Absolute error")
plt.legend()
plt.tight_layout()
plt.savefig(f"{main_name}_delta_absolute_error.pdf")


# FTCS DELTA
results_delta = {}
results_delta_2 = {}
for value in np.linspace(0.00001, 0.00001, 1):
    all_errors = []
    all_true = []
    for price in tqdm(np.linspace(10, 250, 1000)):
        grid_cn, value_1 = fd_cd(N, M, T, price, S_max, K, r, sigma)
        grid_cn, value_2 = fd_cd(N, M, T, price+value, S_max, K, r, sigma)
        error = (abs(disc(value_1)-disc(value_2)))/value
        true_delta = analytical_bs(T, price, K, sigma, r, return_delta=True)
        # print(price, value_1, value_2, true_delta)
        # true_error = error-true_delta
        # true_error = (abs(true_delta-error)/true_delta)
        all_errors.append(error)
        all_true.append(true_delta)
    results_delta[value] = all_errors
    results_delta_2[value] = all_true

for dS in np.linspace(0.00001, 0.00001, 1):
    plt.plot(np.linspace(10, 250, 1000),
             results_delta_2[dS], alpha=0.99, label="True delta")
    plt.plot(np.linspace(10, 250, 1000),
             results_delta[dS], alpha=0.99, label="Simulated delta")
    plt.legend()

plt.xlabel("Stock price")
plt.ylabel("Delta value")
plt.legend()
plt.tight_layout()
plt.savefig(f"{main_name}_true_vs_sim_delta.pdf")
