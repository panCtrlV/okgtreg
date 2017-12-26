# -*- coding: utf-8 -*-
# @Author: Pan Chao
# @Date:   2017-05-12 23:57:19
# @Last Modified by:   Pan Chao
# @Last Modified time: 2017-12-26 11:09:57


import numpy as np
import matplotlib.pyplot as plt

from okgtreg.Kernel import Kernel


kernel = Kernel('gaussian', sigma=0.5)

n = 200
x = np.linspace(0, 1, n)
y = np.repeat(1., n)

G = kernel.gram(x[:, np.newaxis])

beta = (np.linalg.inv(G.T.dot(G))).dot(G.T.dot(y))
y_hat = G.dot(beta)


plt.scatter(x, y_hat)
plt.show()
