"""
An example of group structure detection for IBM job talk.
"""

import pickle, os
import numpy as np
from okgtreg import *
from okgtreg.groupStructureDetection import *


def positivePart(x):
    x[x<0.] = 0.
    return x


def simulateData(n):
    group = Group([1,2], [3,4], [5,6], [7,8], [9,10])
    x = np.random.uniform(0, 2, (n, 10))
    e = np.random.normal(size=n) * 0.1
    y = ( 5. +
          np.sin(x[:,0] * x[:,1]) +
          np.abs(x[:,2] * x[:,3]) +
          x[:,4] ** x[:,5] +
          positivePart(x[:,6] - x[:,7]) +
          x[:,8] / (x[:,9] + 0.1) +
          e ) ** 2
    return Data(y, x), group


nSim = 100
n = 500
# Kernel
kernel = Kernel('gaussian', sigma=0.5)

groups = []
r2s = []

counter = 0
while counter < nSim:
    # Simulation data
    counter += 1
    print("=== Simulation %d ===" % counter)

    np.random.seed(counter)
    data, truegroup = simulateData(n)

    # Group structure detection
    res = splitAndMergeWithRandomInitial2(data, kernel, True, seed=counter)

    groups.append(res.getGroupStructure())
    groups.append(res.r2)


mydir = os.getcwd()
pickle.dump((groups, r2s), open(mydir+'/sim.pkl', 'wb'))
