"""
In this simulation study, we want to evaluate the performance of the
split-and-merge procedure for group structure detection when multiple
covariates are allowed to be randomly split from a group.

We also set a positive threshold for $\Delta R2$ to control the
significance of the improvement of R2 of a OKGT fitting.
"""

import numpy as np

from okgtreg import *
from okgtreg.groupStructureDetection import *


# Simple data simulator
def simpleData(n):
    group = Group([1], [2], [3,4])
    x = np.random.uniform(-1., 1., (n, 4))
    noise = np.random.normal(0., 1., n) * 0.1
    y = x[:,0]**2 + np.sin(x[:, 1]) + x[:, 2] * x[:, 3] + noise
    return Data(y, x), group

# Same kernel
kernel = Kernel('gaussian', sigma=0.5)

# Generate data
np.random.seed(25)
data, group = simpleData(500)

"""
Fitting OKGT using the true group structure:
"""
trueOkgt = OKGTReg(data, kernel=kernel, group=group)
fit = trueOkgt.train('nystroem', 10, 25)
fit['r2']

"""
Group structure detection:

    method: conservative split-and-merge
    split size: 1
    split threshold: 0
    merge threshold: 0.05

Since usually merging two groups results in an improvement in R2,
we set the threshold for merge a little higher than split.
"""
optimalOkgt = splitAndMergeWithRandomInitial2(data, kernel, True, 10, 25,
                                              sThreshold=0., mThreshold=0.05, maxSplit=1)

"""
Systemic simulation
"""

