# -*- coding: utf-8 -*-
# @Author: Pan Chao
# @Date:   2017-05-12 23:57:19
# @Last Modified by:   Pan Chao
# @Last Modified time: 2017-12-26 11:08:47


'''
Use a bivariate Gaussian kernel to recover a univariate function.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from okgtreg.Kernel import Kernel
from okgtreg.Data import Data
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.Group import Group


# Gaussian kernel
kernel = Kernel('gaussian', sigma=0.5)

# Simulate data
p = 2  # number of X
n = 500  # sample size
x = np.random.normal(0, 2, (n, p))
h = np.sin(x[:, 0] * np.pi)   # Only X_1 is relevant
y = h ** 2  # transformation of response

# Fit OKGT with ([1, 2]) as the group structure
data = Data(y, x)
okgt = OKGTReg2(data, kernel=kernel, group=Group([1, 2]))
fit = okgt._train_Vanilla2(h)
print("R2 after fitting OKGT with group structure {group}: {r2}".format(group=Group([1, 2]), r2=fit['r2']))

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 0], data.X[:, 1], fit['f'][:, 0])
plt.show()

# 2D plot
plt.scatter(data.X[:, 1], fit['f'][:, 0])
plt.show()

'''
Indeed, can recover the univariate function by using a
bivariate kernel.
'''
