__author__ = 'panc'

'''
Simulate data from an additive model. Then
fit using additive group structure and the
most general group structure. Check if the
additive group structure results in higher
R2.
'''

import numpy as np

from okgtreg import *


# model 3
# fully additive
# h = 2*x1 + x2**2 + x3**3 + sin(x4*pi) + log(x5+5) + |x6|
# y = ln(h^2)
def model3(n):
    x = np.random.standard_normal((n, p))
    e = np.random.standard_normal((n,)) * 0.01
    h = 2. * x[:, 0] + \
        x[:, 1] ** 2 + \
        x[:, 2] ** 3 + \
        np.sin(x[:, 3] * np.pi) + \
        np.log(np.abs(x[:, 4] + 5.)) + \
        np.abs(x[:, 5]) + \
        e
    y = np.log(h ** 2)
    return Data(y, x), Group([1], [2], [3], [4], [5], [6]), h


# Simulate data
data, group, h = model3(500)

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Fit the additive model
