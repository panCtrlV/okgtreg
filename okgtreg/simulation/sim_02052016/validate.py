__author__ = 'panc'

'''
Use backward stepwise algorithm with penalty for group structure selection.
In order to select the best tuning parameters mu and a, the data is split into
train and test sets. For each pair of (mu, a) in a grid, the backward method
is used on the train set to select a model, then the model is used on the test
set to fit a OKGT (without penalty) and get an estimate of R^2. The optimal pair
of (mu, a) is the one gives the highest R^2.

This is repeated 100 times, then for each pair (mu, a), record the percentage of
time the true group structure is selected.

#
# Important object structures
#
- `trainResults`: {(mu, alpha) : backwardPartition result as a dict}

- `testResults`:  {(mu, alpha) : {group : group_string, test R2 : float value}}

- `groupStructureLookup`: {group_string : test R2}

- `reportResults`:  {group : {(mu, alpha) : test R2} }

#
# Notes on deployment on a server
#
This python script take two command line parameters, model_id and seed number.
An example of command line is:

    python -u validate.py 1 0 > validate-model1-seed0.out

In order to run this script for different MODEL_ID and SEED_NUM on a department's
server, one can simply use `parallel`. Please refer Doug's BITS examples.

In order to deploy parallel execution of the script with different MODEL_ID and SEED_NUM
on a ITAP's RCAC cluster (e.g. Radon), we need some extra work. RCAC uses

1. prepare a

'''

from okgtreg.simulation.sim_02052016.model import selectModel
from okgtreg.Kernel import Kernel
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.groupStructureDetection.backwardPartition import backwardPartitionWithKnownResponse
from okgtreg.simulation.sim_02052016.helper import currentTimestamp

import numpy as np
import sys, os
import pickle
import itertools
from collections import defaultdict

# current time
timestamp = currentTimestamp()

# Parse command line inputs: model id and tuning parameter a
args = sys.argv
model_id = int(args[1])  # model id
seed_num = int(args[2])  # seed for data simulation

# model_id = 1
model = selectModel(model_id)  # Selected model

# Kernel
xKernel = Kernel('gaussian', sigma=0.5)

# Simulation size (repeat for this many data sets)
# nSim = 100

# For each data set, apply one-fold validation for all value
# pairs (50) in the parameter grid
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), 5))  # 5
alphaList = np.arange(1, 11)  # 11

# input: train set, model, mu, alpha, seed
## Simulate a data set (train + test)
ntrain = 500
ntest = 200

np.random.seed(seed_num)
traindata, tgroup, trian_g = model(ntrain)
testdata, testgroup, test_g = model(ntest)

## results container
trainResults = {}
testResults = {}
groupStructureLookup = defaultdict(list)

counter = 0
for (mu, alpha) in itertools.product(muList, alphaList):  # quick way to get cartesian pairs
    counter += 1
    print("#######################################################")
    print("# %d : mu = %.10f, alpha = %.02f" % (counter, mu, alpha))
    print("#######################################################")
    # The true response transformation is given (g).
    trainRes = backwardPartitionWithKnownResponse(traindata, xKernel, trian_g,
                                                  mu=mu, alpha=alpha)
    # There are likely duplicated group structures for different (mu, a)
    group = trainRes['group']
    groupStr = group.__str__()
    trainR2 = trainRes['r2']
    # Collect selection result on training data for each (mu, alpha) pair
    trainResults[(mu, alpha)] = trainRes
    if groupStr in groupStructureLookup.keys():
        # Different (mu, alpha) may select the same group structures.
        # In this case, no need to repeatedly fit the same group structure
        # on the test data, just add the same R2 in testResults.
        testResults[(mu, alpha)] = dict(r2=groupStructureLookup[groupStr], group=groupStr)
    else:
        # If the group structure selected from the training set is new,
        # fit OKGT using the group structure on the test data.
        testOKGT = OKGTReg2(testdata, kernel=xKernel, group=group)
        # Fit with the true response transformation (g)
        testRes = testOKGT._train_Vanilla2(test_g)
        groupStructureLookup[group.__str__()] = testRes['r2']
        testResults[(mu, alpha)] = dict(r2=testRes['r2'], group=groupStr)

# bestParameters = max(testResults, key=testResults.get)
# print bestParameters

# Different (mu, a) can have same group structures,
# thus the same test R2. We want the (mu, a) pairs (
# and the corresponding group structure) that gives
# the largest R2
reportResults = defaultdict(dict)
for k, v in testResults.iteritems():
    reportResults[v['group']][k] = v['r2']

# Object to pickle.dump
dumpObject = dict(train=trainResults,
                  test=testResults,
                  lookup=groupStructureLookup,
                  report=reportResults)

# Pickle results
dirpath = os.getcwd()
filename, file_extension = os.path.splitext(__file__)
filename = filename + \
           "-model" + str(model_id) + \
           "-seed" + str(seed_num) + \
           "-" + timestamp + ".pkl"
saveto = dirpath + '/' + filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)


# --------------------------------------------
# # Create commands for parallel batch jobs
# for seed in range(100):
#     for model_id in np.arange(1)+1:
#         print("python -u validate.py %d %d > validate-model%d-seed%d-%s.out" %
#               (model_id, seed, model_id, seed, timestamp))
# --------------------------------------------


# if __name__=='__main__':
#     model_id = 1
#     dirpath = os.getcwd()
#     filename, file_extension = os.path.splitext(__file__)
#     filename = filename + "-model" + str(model_id) + "-sim1.pkl"
#     saveto = dirpath + '/' + filename
#
#     print saveto



# ---------------------------------
# x = {1:{'a':1, 'b':2},
#      2:{'a':1, 'b':3},
#      3:{'a':2, 'b':2},
#      4:{'a':1, 'b':5},
#      5:{'a':2, 'b':8}}

# [x[k] for k in [k for k,v in x.iteritems() if v['a']==1]]

# # Find all keys share the same value
# x1List = [(k, v['a']) for k,v in x.iteritems()]
# x1 = dict(x1List)
# x1
#
# from collections import defaultdict
#
# new_dict = defaultdict(list)
# for k, v in x1.iteritems():
#     new_dict[v].append(k)
#
# new_dict

# ------------------------------------
# with open("okgtreg/simulation/sim_02052016/validate-model1-sim1-201602062345.pkl", 'rb') as f:
#     res = pickle.load(f)
