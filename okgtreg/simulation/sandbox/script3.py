__author__ = 'panc'

'''
Use a bivariate Gaussian kernel to recover
a univariate function.
'''

import numpy as np

from okgtreg.Kernel import Kernel
from okgtreg.Data import Data
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.Group import Group

# Gaussian kernel
kernel = Kernel('gaussian', sigma=0.5)

# Simulate data
p = 2
n = 500
x = np.random.normal(0, 2, (n, p))
h = np.sin(x[:, 0] * np.pi)
y = h ** 2

data = Data(y, x)
okgt = OKGTReg2(data, kernel=kernel, group=Group([1, 2]))
fit = okgt._train_Vanilla2(h)
print fit['r2']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.X[:, 0], data.X[:, 1], fit['f'][:, 0])

'''
Indeed, can recover the univariate function by using a
bivariate kernel.
'''
