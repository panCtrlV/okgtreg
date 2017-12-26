__author__ = 'panc'

'''
True model used for group structure selection
'''

import numpy as np

from okgtreg import *


# 15 covariates are evenly partitioned into five groups (each has 3 covariates)
# A group of three covariates follows a mixed multi-variate normal distribution.
def simulate_data(n):
    # n = 500
    p = 15
    d = 5  # number of groups
    g = p / d  # number of covariates for each group
    h = 0.01 * np.sqrt(g)

    x = np.random.standard_normal((n, p))
    # Mean vectors
    mu1 = np.array([-3., -3., -3.])
    mu2 = np.array([0., 0., 0.])
    mu3 = np.array([3., 3., 3.])
    mu = [mu1, mu2, mu3]
    # Covariance matrices
    cov1 = np.array([[1., 0.8, 0.8], [0.8, 1., 0.8], [0.8, 0.8, 1.]])  # Equilateral triangle
    cov2 = np.array([[1., 0.1, 0.1], [0.1, 1., 0.8], [0.1, 0.8, 1.]])  # Isosceles triangle
    cov3 = np.array([[1., -0.5, -0.1], [-0.5, 1., 0.2], [-0.1, 0.2, 1.]])
    cov = [cov1, cov2, cov3]
    L1 = np.linalg.cholesky(cov1)
    L2 = np.linalg.cholesky(cov2)
    L3 = np.linalg.cholesky(cov3)
    L = [L1, L2, L3]
    # Mixing probability vectors
    w1 = np.linspace(0.1, 0.9, 5)
    w2 = (1 - w1) * 0.3
    w3 = 1. - w1 - w2
    w = np.column_stack([w1, w2, w3])
    # Synthesize data
    data = []
    for l in range(d):  # for each group
        datal = []
        for i in range(3):  # for each mixing component
            mnorm_x = x[:, l * g: (l + 1) * g].dot(L[i].T) + mu[i]  # multivariate normal
            groupi = np.exp(np.linalg.norm(mnorm_x, ord=2, axis=1))
            datal.append(groupi)
        data.append(np.column_stack(datal).dot(w[l, :]))
    y = np.log(np.log(np.column_stack(data)).sum(axis=1))

    return Data(y, x), Group([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(25)
    data, group = simulate_data(500)
    plt.hist(data.y, 30)
