from analytical_bs import analytical_bs
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

# Parameters
r = 0.04
sigma = 0.3
S_max = 200
K = 110
T = 1
N = 50#100
M = 99 #100

midpoint = int(M/2)

S_0 = 100 # In the money
# S_0 = 110 # At the money
# S_0 = 120 # Out of the money

grid_cn, value = fd_cd(N, M, T, S_0, S_max, K, r, sigma)

all_S = np.linspace(0, S_max, M+1)
price_array_cn = []
for array in grid_cn:
    price_array_cn.append(array[midpoint])

grid_ftcs, value = ftcs(N, M, T, S_0, S_max, K, r, sigma)

price_array_ftcs = []
for array in grid_ftcs:
    price_array_ftcs.append(array[midpoint])

price_array_an = []
for price in all_S:
    price_array_an.append(analytical_bs(T=T, S0=price,K=K,sigma=sigma,r=r))

# ----- 2D price plot ------
plt.plot(all_S, price_array_cn, label="CN", lw=3)
plt.plot(all_S, price_array_an, label="BS", ls=":",lw=2)
plt.plot(all_S, price_array_ftcs, label="FTCS", ls='--', lw=3)
plt.legend()
plt.ylabel("Option price")
plt.xlabel("Stock price")


# ----- 3D final grid plot ------
# 3D CN
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
all_S = all_S[::-1] # THE ACTUAL DATA SHOWS THE REVERSED AXIS BUT THE AXIS LABEL DOES NOT, COMMENT THIS OUT TO SEE TRUE GRAPH
X = all_S
Y = np.arange(N)
X, Y = np.meshgrid(X, Y)
Z = grid_cn[:,:-1]
Z = Z.transpose()

print(X.shape, Y.shape, Z.shape)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
ax.set_xlabel(xlabel="Stock price")
ax.set_ylabel(ylabel="Time spaces")
ax.set_zlabel(zlabel="Option price")
ax.set_title("CN")
# ax.invert_zaxis()
plt.show()


# 3D FTCS
fig = plt.figure()
ax = fig.gca(projection='3d')

# THE ACTUAL DATA SHOWS THE REVERSED AXIS BUT THE AXIS LABEL DOES NOT, COMMENT THIS OUT TO SEE TRUE GRAPH
all_S = all_S[::-1]
X = all_S
Y = np.arange(N)
X, Y = np.meshgrid(X, Y)
Z = grid_ftcs[:, :-1]
Z = Z.transpose()

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
ax.set_xlabel(xlabel="Stock price")
ax.set_ylabel(ylabel="Time spaces")
ax.set_zlabel(zlabel="Option price")
ax.set_title("FTCS")
# ax.invert_zaxis()
plt.show()


# ----- Plot error (vs analytical) +  try varying N and M ------
errors_cn = []
errors_ftcs = []
for index in range(len(price_array_an)):
    errors_cn.append(abs(price_array_cn[index] - price_array_an[index]))
    errors_ftcs.append(abs(price_array_ftcs[index] - price_array_an[index]))


# for _ in range(len(errors_cn)):
plt.plot(all_S, errors_cn, label="CN", lw=3)
plt.plot(all_S, errors_ftcs, label="BS", ls=":", lw=2)



# ----- Error depending on grid size -----
N = 50
results = {}
grid_sizes = np.linspace(10, 700, 150) # try 5,500,20
moneyness = [100]
for money in moneyness:
    abort = True
    s0 = money
    all_prices_cn = []
    all_prices_ftcs = []
    all_errors_cn = []
    all_errors_ftcs = []
    for grid_size in tqdm(grid_sizes):
        bs = analytical_bs(T=T, S0=money, K=K, sigma=sigma, r=r)
        grid_cn, value_1 = fd_cd(N, int(grid_size), T, S_0, S_max, K, r, sigma)
        if abort:
            grid_ftcs, value_2=ftcs(N, int(grid_size), T, S_0, S_max, K, r, sigma)
            if value_2 == 0.0:
                abort = False
        else:
            value_2 = bs
        # Append prices
        all_prices_cn.append(value_1)
        all_prices_ftcs.append(value_2)
        all_errors_cn.append(abs(value_1-bs)/bs)
        all_errors_ftcs.append(abs(value_2-bs)/bs)
    results[money] = [all_prices_cn, all_prices_ftcs, all_errors_cn, all_errors_ftcs]
results


e_1 = results[100][2]
e_2 = results[100][3]
# plt.yscale("log")
for value in e_2:
    if value == 0:
        index = e_2.index(value)
        index -= 4

first = e_2[0:index]
second = e_2[index:index+3]
plt.ylim((0,0.5))
plt.plot(grid_sizes, e_1, label="CN")
plt.plot(list(grid_sizes)[0:35], e_2[0:35], color="tab:red", ls='--')
plt.plot(list(grid_sizes)[0:index], first, color="tab:orange")



# plt.plot(grid_sizes, e_2)
plt.legend()

# ---- Plot GREEKS as of S! Use Menno -> fast
# double call_delta_fdm(const double S, const double K, const double r, const double v, const double T, const double delta_S) {
#     return (call_price(S + delta_S, K, r, v, T) - call_price(S, K, r, v, T))/delta_S;

def disc(value):
    return np.exp(-r) * value

def delta(S, dS):
    grid_cn, value_1 = fd_cd(N, M, T, S, S_max, K, r, sigma)
    grid_cn, value_2 = fd_cd(N, M, T, S+dS, S_max, K, r, sigma)
    # return ((np.exp(value_1) - np.exp(value_2))/dS) *  np.exp(-r)
    return (abs(disc(value_1)-disc(value_2)))/dS

delta = delta(price, 0.00001)

bs = analytical_bs(T=T, S0=price, K=K, sigma=sigma, r=r, delta=True)

print(delta)
print(bs)
print((delta - bs)/bs)

for price in np.linspace()
