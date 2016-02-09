__author__ = 'panc'

import numpy as np
from okgtreg.Kernel import Kernel

kernel = Kernel('gaussian', sigma=0.5)

n = 200
x = np.linspace(0, 1, n)
y = np.repeat(1., n)

G = kernel.gram(x[:, np.newaxis])

beta = (np.linalg.inv(G.T.dot(G))).dot(G.T.dot(y))
y_hat = G.dot(beta)

import matplotlib.pyplot as plt

plt.scatter(x, y_hat)
