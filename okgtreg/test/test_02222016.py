__author__ = 'panc'

import numpy as np

from okgtreg.Kernel import Kernel

kernel = Kernel('gaussian', sigma=0.5)

# mapping at a scalar value
phi = kernel.kernelMapping(0.5)
phi(0.5)

# mapping at a vector
phi = kernel.kernelMapping(np.array([0.5, 0.5]))
phi(np.array([0.5, 0.5]))
phi(np.array([0.5, 1.]))

# kernel span
coef = np.array([0.1, 3., -0.5])
x = np.array([[0, .5, .3],
              [-0.3, 0.4, 0.2],
              [0.4, 0.9, 1.]])
f = kernel.kernelSpan(x, coef)
y = np.array([0, .5, .3])
f(y)
