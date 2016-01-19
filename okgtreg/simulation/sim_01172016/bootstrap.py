__author__ = 'panc'

'''
This study is a sequel of that in "script.py". It performs a
bootstrap to estimate the variances / standard deviations for
the R^2 differences between different group structures.

The standard deviations are used to test the significance of
those differences.

For the details of parsing commend line arguments for Python
scripts, please refer to the tutorial:

    http://www.tutorialspoint.com/python/python_command_line_arguments.htm


'''

import sys, os
import random
import numpy as np
import pickle

from okgtreg.simulation.sim_01172016.model import *
from okgtreg.simulation.sim_01172016.helper import *
from okgtreg import *

# Parse command line arguments
args = sys.argv
bootstrapSeed = int(args[1])  # an integer

print("=== Bootstrap seed: %d ===" % bootstrapSeed)

# The simulated data
n = 500
np.random.seed(25)
data, truegroup = simpleData(n)

# Bootstrap (sample with replacement)
rg = random.Random(bootstrapSeed)
sampled_ids = []
for i in range(n):
    sampled_ids.append(rg.choice(range(n)))

bootstrapData = Data(data.y[sampled_ids], data.X[sampled_ids])

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Fit OKGT for all possible group structures
bootstrapRes = {}

allpartitions = list(partitions(set(range(1, truegroup.p + 1))))
allpartitions = [tuple(list(item) for item in group) for group in allpartitions]

for i in range(len(allpartitions)):
    group = Group(*allpartitions[i])
    okgt = OKGTReg(bootstrapData, kernel=kernel, group=group)
    fit = okgt.train()  # vanilla train
    r2 = fit['r2']
    bootstrapRes[group] = r2
    print("%d : %s : %.10f" % (i + 1, group, r2))

# Save results
mydir = os.getcwd()
saveto = mydir + '/bootstrap/' + __file__ + '-' + str(bootstrapSeed) + '.pkl'
# print saveto
pickle.dump(bootstrapRes, open(saveto, 'wb'))

# # Generate bash script
# for i in range(100):
#     print("python -u bootstrap.py %d > bootstrap/bootstrap.py.out.%d" % (i + 1, i + 1))
