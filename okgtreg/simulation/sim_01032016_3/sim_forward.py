__author__ = 'panc'

"""
This is similar to the corresponding simulation in "sim_01032016_2".
We simulate data from the same model but with **independent x**.
"""

import os
import pickle
import numpy as np
from sklearn import preprocessing
from okgtreg import *
from okgtreg.groupStructureDetection import *
from okgtreg.simulation.sim_01032016_3.model import simulateData

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
    # Normalize data
    data.y = preprocessing.scale(data.y)
    data.X = preprocessing.scale(data.X)

    # Group structure detection
    # res = splitAndMergeWithRandomInitial2(data, kernel, True, seed=counter)
    res = forwardSelection(data, kernel, 'nystroem', seed=counter)
    # res = backwardPartition(data, kernel, 'nystroem', seed=counter)

    groups.append(res['group'])
    r2s.append(res['r2'])
    print('\n')

# Pickle results
mydir = os.getcwd()
pickle.dump((groups, r2s), open(mydir + '/' + __file__ + '.pkl', 'wb'))
