__author__ = 'panc'

"""
This simulation is intended to be used for IBM jab talk.

The purpose of this simulation is to demonstrate the usefulness of the
**backward partition** group detection procedure to find an optimal group
structure given a data set.

Ideally, the procedure should recover the true group structure in the
simulation.

We employ the same mode as that in "sim_01022016". However, this time the
observations are simulated such that each pair of covariates in a group
are correlated.
"""
import os
import pickle
import numpy as np
from okgtreg import *
from okgtreg.groupStructureDetection import *


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
    # res = splitAndMergeWithRandomInitial2(data, kernel, True, seed=counter)
    # res = forwardSelection(data, kernel, 'nystroem', seed=counter)
    res = backwardPartition(data, kernel, 'nystroem', seed=counter)

    groups.append(res['group'])
    r2s.append(res['r2'])
    print('\n')

# Pickle results
mydir = os.getcwd()
pickle.dump((groups, r2s), open(mydir+'/'+__file__+'/'+'.pkl', 'wb'))
