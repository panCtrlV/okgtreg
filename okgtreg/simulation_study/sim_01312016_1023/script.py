__author__ = 'panc'

'''
Using 1-fold cross-validation for backward group structure determination
with penalized OKGT.
'''

import os, sys
import pickle
import numpy as np
import datetime

from okgtreg import *
from okgtreg.simulation.sim_01312016_1023.model import simulate_data
from okgtreg.groupStructureDetection.backwardPartition import backwardPartition

currentTimeStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

args = sys.argv
a = float(args[1])

# Values of tuning parameters $\lambda$ and $a$
lmbda = 1e-4

# Simulate data
n = 1000
np.random.seed(25)
data, group = simulate_data(n)

# Split into training and testing
ntrain = int(0.8 * n)
traindata = Data(data.y[:ntrain], data.X[:ntrain, :])
testdata = Data(data.y[ntrain:], data.X[ntrain:, :])

print("\tTraining size = %d" % ntrain)
print("\tTesting size = %d" % (n - ntrain,))

# Fix kernel
print("Kernel: Gaussian (0.5)")
kernel = Kernel('gaussian', sigma=0.5)

# Print simulation information
s1 = "# Simulation time: %s" % currentTimeStr
s2 = "# Tuning parameters: lambda = %.10f, a = %.02f" % (lmbda, a)
s3 = "# Kernel: Gaussian (0.5)"
s4 = "# Training method: vanilla"
s5 = "# Selection method: backward"
s6 = "# Sample size: %d, training size: %d, testing size: %d" % (n, ntrain, n - ntrain)
swidth = max([len(s) for s in [s1, s2, s3, s4, s5, s6]]) + 2
s0 = ''.join(np.repeat("#", swidth))
s7 = ''.join(np.repeat("#", swidth))
print '\n'.join((s0, s1, s2, s3, s4, s5, s6, s7))

# Cross-validation for group structure selection
## group structure selection on training set
selection_res = backwardPartition(traindata, kernel, lmbda=lmbda, a=a)
selectedGroup = selection_res['group']
okgt_train = OKGTReg2(traindata, kernel=kernel, group=selectedGroup)
r2_train = okgt_train.train()
## validate on testing set
okgt_test = OKGTReg2(testdata, kernel=kernel, group=selectedGroup)
fit = okgt_test.train()
validateR2 = fit['r2']

# validateR2 = []
# selectedGroupStructures = []
# for a in range(len(a)):
#     # group structure selection on training set
#     selection_res = backwardPartition(traindata, kernel, lmbda=lmbda, a=a)
#     selectedGroup = selection_res['group']
#     selectedGroupStructures.append(selectedGroup)
#     # validate on testing set
#     okgt = OKGTReg2(data, kernel=kernel, group=selectedGroup)
#     fit = okgt.train()
#     validateR2.append(fit['r2'])
#
# bestIndex = np.argmin(validateR2)
# bestGroupStructure = selectedGroupStructures[bestIndex]

# Pickle results
mydir = os.getcwd()
filename = "script-a-%.01f.pkl" % a
saveto = mydir + '/' + filename
pickle.dump((), open(saveto, 'wb'))

# # Create shell script
# for a in range(1, 11):
#     print("python -u script.py %d > script-a-%d.out" % (a, a))

# # Create shell script for a in [2, 2.9]
# for a in np.linspace(2, 2.9, 10):
#     print("python -u script.py %.01f > script-a-%.01f.out" % (a, a))

# # Create shell script for a in [3.1, 3.9]
# for a in np.linspace(3.1, 3.9, 9):
#     print("python -u script.py %.01f > script-a-%.01f.out" % (a, a))
