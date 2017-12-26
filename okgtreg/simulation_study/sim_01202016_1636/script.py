__author__ = 'panc'

import sys, os
import random
import numpy as np
import pickle

from okgtreg.simulation.sim_01202016_1636.models import *
from okgtreg.simulation.sim_01202016_1636.helper import *
from okgtreg import *

# Parse command line arguments
args = sys.argv
model_id = int(args[1])  # model index 1-10

# Selected model
model = selectModel(model_id)

print model

print("=== Selected Model: %d ===" % model_id)

# Simulate data from the model
n = 500
np.random.seed(25)
data, truegroup = model(n)

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Fit OKGT for each possible group structure
res = {}

## all group structures
allpartitions = list(partitions(set(range(1, 7))))
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
filename = 'script-model-' + str(model_id) + '.pkl'
saveto = mydir + '/' + filename
pickle.dump(res, open(saveto, 'wb'))

# # Create bash script
# for i in range(10):
#     print("python -u script.py %d > script-model-%d.out" % (i + 1, i + 1))
