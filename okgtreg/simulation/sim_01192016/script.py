__author__ = 'panc'

'''
Run OKGT for all possible group structures for the data
simulated from the model in this folder. The model uses
more complex transformations than those in the simulation
study "sim_01172016".

Since the number of covariates is the same as that in
"sim_01172016", the total number of group structures is
still 203.

We still use the "vanilla" method for OKGT training.
'''

from okgtreg.simulation.sim_01192016.model import *
from okgtreg.simulation.sim_01192016.helper import *
from okgtreg import *

import numpy as np
import pickle
import os

# Simulate data from a new simple model
n = 500
np.random.seed(25)
data, truegroup = simpleData_01192016(n)

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Fit OKGT for each possible group structure
res = {}

## all group structures
allpartitions = list(partitions(set(range(1, truegroup.p + 1))))
allpartitions = [tuple(list(item) for item in group) for group in allpartitions]

## fit okgt for all group structures
for i in range(len(allpartitions)):
    group = Group(*allpartitions[i])
    okgt = OKGTReg(data, kernel=kernel, group=group)
    fit = okgt.train()  # vanilla train
    r2 = fit['r2']
    res[group.__str__()] = r2
    print("%d : %s : %.10f" % (i + 1, group.__str__(), r2))

# Save results
mydir = os.getcwd()
saveto = mydir + '/' + __file__ + '.pkl'
pickle.dump(res, open(saveto, 'wb'))


# group = Group([1], [2,3], [4,5,6])
# okgt = OKGTReg(data, kernel=kernel, group=group)
# fit = okgt.train()
# fit['r2']
