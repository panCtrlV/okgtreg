"""
Debug split-and-merge
"""

import numpy as np

from okgtreg import *
from okgtreg.groupStructureDetection import *

n = 500
kernel = Kernel('gaussian', sigma=0.5)

val = 0.01

np.random.seed(25)
data, group = DataSimulator.SimData_Wang04WithTwoBivariateGroups(n)

optimalOkgt = splitAndMergeWithRandomInitial2(data, kernel, True, 10, 25,
                                              sThreshold=0., mThreshold=val, maxSplit=1)