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
