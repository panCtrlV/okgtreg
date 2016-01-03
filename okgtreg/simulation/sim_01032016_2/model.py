import numpy as np
from okgtreg import *


def positivePart(x):
    x[x<0.] = 0.
    return x

# The first covariate in each group are independent Unif(0,2),
# and the second covariate in a group is created by multiplying a
# constant to the first covariate.
def simulateData(n):
    group = Group([1,2], [3,4], [5,6], [7,8], [9,10])
    x1 = np.random.uniform(0, 2, (n, 5))
    # x2 = x1 + np.random.normal(size=(n,5)) * 0.1
    x2 = x1 * 2.
    x = np.vstack([np.vstack([x1[:,i], x2[:,i]]) for i in range(5)]).T
    e = np.random.normal(size=n) * 0.1
    y = ( 5. +
          np.sin(x[:,0] * x[:,1]) +
          np.abs(x[:,2] * x[:,3]) +
          x[:,4] ** x[:,5] +
          positivePart(x[:,6] - x[:,7]) +
          x[:,8] / (x[:,9] + 0.1) +
          e ) ** 2
    return Data(y, x), group