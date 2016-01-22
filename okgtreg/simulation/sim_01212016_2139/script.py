__author__ = 'panc'

import numpy as np
import pickle
import sys, os

from okgtreg import *
from okgtreg.simulation.sim_01212016_2139.model import simplePolyModel
from okgtreg.simulation.sim_01212016_2139.helper import partitions

# Kernel
kernel = Kernel('polynomial', degree=2)
print kernel

# Simulate data from the model
n = 500
np.random.seed(25)
data, truegroup = simplePolyModel(500)

kernel.gram(data.y[:, np.newaxis])


# Fit OKGT for each possible group structure
print("=== Kernel: Polynomial (intercept: 0, slope: 1, degree: 2) ===")
res = {}

## all group structures
allpartitions = list(partitions(set(range(1, truegroup.p + 1))))
allpartitions = [tuple(list(item) for item in group) for group in allpartitions]

for i in range(len(allpartitions)):
    group = Group(*allpartitions[i])
    okgt = OKGTReg(data, kernel=kernel, group=group)
    fit = okgt.train()  # vanilla train
    r2 = fit['r2']
    res[group] = r2
    print("%d : %s : %.10f" % (i + 1, group, r2))

# Save results
mydir = os.getcwd()
filename = __file__ + '.pkl'
saveto = mydir + '/' + filename
pickle.dump(res, open(saveto, 'wb'))
