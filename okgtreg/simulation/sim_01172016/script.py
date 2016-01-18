__author__ = 'panc'

'''
By using a simple / small model, fit OKGT for all possible
group structures. This simulation is used to investigate if
group structures have any ordering.

Note: 6 covariates have 203 different partitions. In general,
the number of partitions for n variables is given by the Bell
number.
'''

from okgtreg.simulation.sim_01172016.model import *
from okgtreg.simulation.sim_01172016.helper import *
from okgtreg import *

import numpy as np
import pickle
import os

# Simulate data from a simple model
n = 500
np.random.seed(25)
data, truegroup = simpleData(n)

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Fit OKGT for each possible group structure
res = {}

allpartitions = list(partitions(set(range(1, 7))))
allpartitions = [tuple(list(item) for item in group) for group in allpartitions]

# for i in range(len(allpartitions)):
for i in range(10):
    group = Group(*allpartitions[i])
    okgt = OKGTReg(data, kernel=kernel, group=group)
    fit = okgt.train()  # vanilla train
    r2 = fit['r2']
    # res.append((group, r2))
    res[group] = r2
    print("%d : %s : %.10f" % (i + 1, group, r2))

# Save results
mydir = os.getcwd()
pickle.dump(res, open(mydir + '/' + __file__ + '.pkl', 'wb'))


# # Sort results by r2
# import operator
#
# sorted(res.items(), key=operator.itemgetter(1), reverse=True)
