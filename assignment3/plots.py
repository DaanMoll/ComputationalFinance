import matplotlib.pyplot as plt
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
T =1
N = 1000
M = 100


S_0 = 100 # In the money
S_0 = 110 # At the money
S_0 = 120 # Out of the money

grid_cn, value = fd_cd(N, M, T, S_0, S_max, K, r, sigma)

all_S = np.linspace(0, S_max, M+1)
price_array_cn = []
for array in grid_cn:
    price_array_cn.append(array[50])

grid_ftcs, value = ftcs(N, M, T, S_0, S_max, K, r, sigma)

price_array_ftcs = []
for array in grid_ftcs:
    price_array_ftcs.append(array[50])

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
