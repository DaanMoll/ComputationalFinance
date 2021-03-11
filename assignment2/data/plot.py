import matplotlib.pyplot as plt
import pickle
from numpy.lib.function_base import append
import seaborn as sns
import pandas as pd
import numpy as np

with open('/Users/adrianfuertes/Google Drive/UvA - CLS/Year 1/Semester 1/Period 3/ABM/hub/ComputationalFinance/assignment2/data/bump.pickle', 'rb') as handle:
    b = pickle.load(handle)

with open('/Users/adrianfuertes/Google Drive/UvA - CLS/Year 1/Semester 1/Period 3/ABM/hub/ComputationalFinance/assignment2/data/likelihood.pickle', 'rb') as handle:
    l = pickle.load(handle)


anal = 0.018206369779490493


for 
# data = []
# for key in l.keys():
#     print(np.std(l[key][1]))
#     # df["Absolute Error"] = l[key][1]
#     # df["Path number"] = key
#     data.append(l[key][1])

# data = data[1:-1]
# sns.boxplot(data=data)


# data = []
# for key in l.keys():
#     print(np.std(l[key][1]))
#     # df["Absolute Error"] = l[key][1]
#     # df["Path number"] = key
#     data.append(b[key][1])

# colors = {
#     1: "green",
#     0.5: "blue",
#     0.01: "red",
#     0.001: "purple",
#     0.0001: "orange"
# }


# a = []
# t = []
# c = []
# r = []
# for epsilon in b.keys():
#     for key in b[epsilon].keys():
#         if len(a) < 5:
#             a.append(np.mean(b[epsilon][key][1]))
#         elif len(t) < 5:
#             t.append(np.mean(b[epsilon][key][1]))
#         elif len(c) < 5:
#             c.append(np.mean(b[epsilon][key][1]))
#         elif len(r) < 5:
#             r.append(np.mean(b[epsilon][key][1]))

# plt.plot(keys, c)
# plt.plot(keys, r)
# plt.plot(keys, t)
# plt.plot(keys, a)


# keys = []
# for key in b[epsilon].keys():
#     keys.append(key)

