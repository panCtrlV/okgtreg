__author__ = 'panc'

from okgtreg.simulation.sim_02042016.model import *
from okgtreg.simulation.sim_02042016.helper import partitions, rkhsCapacity
from okgtreg import *

import numpy as np
import sys, os
import datetime
import pickle

'''
Exhaustive okgt training, no penalty
'''

# Parse command line inputs: model id and tuning parameter a
args = sys.argv
model_id = int(args[1])  # model id
model = selectModel(model_id)  # Selected model
# a = float(args[2]) # tuning parameter a

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Print simulation information
currentTimeStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
s1 = "# Exhaustive OKGT fitting"
s2 = "# Start time: %s" % currentTimeStr
s3 = "# Selected model: %d" % model_id
s4 = "# Kernel: Gaussian (0.5)"
s5 = "# Training method: vanilla"
s6 = "# Sample size: 500"
swidth = max([len(s) for s in [s1, s2, s3, s4, s5, s6]]) + 2
s0 = ''.join(np.repeat("#", swidth))
s7 = ''.join(np.repeat("#", swidth))
print '\n'.join((s0, s1, s2, s3, s4, s5, s6, s7))

# Simulate data
n = 500
np.random.seed(25)
data, tgroup = model(n)

# Train regular OKGT to obtain R^2 for all group structures,
# then for each combination of (mu, a), calculated the penalized R^2

## all group structures
allpartitions = list(partitions(set(range(1, 7))))
allpartitions = [tuple(list(item) for item in group) for group in allpartitions]

## Train OKGT for all group structures (eps=1e-6)
res = {}
for i in range(len(allpartitions)):
    group = Group(*allpartitions[i])
    okgt = OKGTReg2(data, kernel=kernel, group=group)
    fit = okgt.train()  # vanilla train
    r2 = fit['r2']
    res[group] = r2
    print("%d : %s : %.10f" % (i + 1, group, r2))

# Pickle results
mydir = os.getcwd()
filename = 'script-model-' + str(model_id) + '.pkl'
saveto = mydir + '/' + filename
pickle.dump(res, open(saveto, 'wb'))
