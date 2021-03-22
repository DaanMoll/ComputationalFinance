from scipy.stats import norm
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def ftcs_menno(N, M, T, S_0, S_max, K, r, sigma):

