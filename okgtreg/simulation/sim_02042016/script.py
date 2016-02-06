__author__ = 'panc'

from okgtreg.simulation.sim_02042016.model import *
from okgtreg.simulation.sim_02042016.helper import partitions, rkhsCapacity
from okgtreg import *

import numpy as np
import sys, os
import datetime
import pickle

'''
Exhaustive okgt training, no penalty.
100 repetition for each model, so that rank frequency
can be calculated.
'''

# Parse command line inputs: model id and tuning parameter a
args = sys.argv
model_id = int(args[1])  # model id
model = selectModel(model_id)  # Selected model
# a = float(args[2]) # tuning parameter a

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Simulation size
nSim = 100


# Print simulation information
currentTimeStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
s1 = "# Exhaustive OKGT fitting"
s2 = "# Start time: %s" % currentTimeStr
s3 = "# Selected model: %d" % model_id
s4 = "# Kernel: Gaussian (0.5)"
s5 = "# Training method: vanilla"
s6 = "# Sample size: 500"
s7 = "# Simulation size (repetition): %d" % nSim
swidth = max([len(s) for s in [s1, s2, s3, s4, s5, s6, s7]]) + 2
s0 = ''.join(np.repeat("#", swidth))
s8 = ''.join(np.repeat("#", swidth))
print '\n'.join((s0, s1, s2, s3, s4, s5, s6, s7, s8))


# Train regular OKGT to obtain R^2 for all group structures,
# then for each combination of (mu, a), calculated the penalized R^2

## all group structures
allpartitions = list(partitions(set(range(1, 7))))
allpartitions = [tuple(list(item) for item in group) for group in allpartitions]

res_all = []
for i in range(nSim):
    print("=== seed: %d ===" % i)

    # Simulate data
    n = 500
    np.random.seed(i)
    data = model(n)[0]  # only need data

    ## Train OKGT for all group structures (eps=1e-6)
    res = {}
    for i in range(len(allpartitions)):
        group = Group(*allpartitions[i])
        okgt = OKGTReg2(data, kernel=kernel, group=group)
        fit = okgt.train()  # vanilla train
        r2 = fit['r2']
        res[group] = r2
        print("%d : %s : %.10f" % (i + 1, group, r2))
    res_all.append(res)

# Pickle results for a model
mydir = os.getcwd()
filename = 'script-model-' + str(model_id) + '-sim100.pkl'
saveto = mydir + '/' + filename
pickle.dump(res_all, open(saveto, 'wb'))
