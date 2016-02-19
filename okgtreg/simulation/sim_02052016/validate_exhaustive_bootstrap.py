__author__ = 'panc'

'''
Because using one-fold validation (train + test) approach for
selecting the best tuning parameters was not successful, we add
a bootstrap step in the algorithm.

In particular, for each (mu, alpha) pair, bootstrap is performed
on the training set so that we can (hopefully) estimate the stand-
ard deviation of the test R2. Then we use the quantity

    mean of R2 - 0.1 * std.dev of R2

as the criterion to select the best parameters.
'''

import numpy as np
import sys, os
from collections import defaultdict
import itertools
import pickle

from okgtreg.groupStructureDetection.backwardPartition import rkhsCapacity
from okgtreg.simulation.sim_02052016.helper import currentTimestamp
from okgtreg.simulation.sim_02052016.model import selectModel
from okgtreg.Kernel import Kernel
from okgtreg.Group import Group
from okgtreg.utility import partitions
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.Data import Data

#########################
# Simulation Parameters #
#########################

# For each data set, apply one-fold validation for all value
# pairs (25) in the parameter grid for each possible group
# structures (203 different structures for 6 covariates)
mu_size = 5
alpha_size = 10

muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
alphaList = np.arange(1, alpha_size + 1)

allPartitions = [tuple(list(item) for item in p) for p in partitions(set(np.arange(6) + 1))]
allGroupStructures = [Group(*p) for p in allPartitions]

# Parse command line inputs: model id & tuning parameter a
args = sys.argv
model_id = int(args[1])  # model id
data_seed = int(args[2])  # seed for data simulation
bt_seed = int(args[3])  # bootstrap seed

# selected model
model = selectModel(model_id)  # Selected model

# Kernel:
# The response transformation is assumed to be known,
# so we only need to transform the covariates.
xKernel = Kernel('gaussian', sigma=0.5)

# Print INFO
timestamp = currentTimestamp()  # current time
print "###############################################"
print "# Model %d" % model_id
print "# Seed = %d" % data_seed
print "# Kernel: Gaussian (0.5). Response fixed."
print "# Mu values: ", muList
print "# Alpha values: ", alphaList
print "###############################################"

######################################
# Simulate a data set (train + test) #
######################################
ntrain = 500
ntest = 500

np.random.seed(data_seed)
traindata, tgroup, train_g = model(ntrain)
testdata, testgroup, test_g = model(ntest)

##################
# Training Phase #
##################
# nbootstrap = 100
# bootstrap_seeds = range(nbootstrap)

# bootstrap_dict = defaultdict(list)  # (mu, alpha) : [all bootstrap test r2's]
bootstrap_dict = {}
testedGroupStructures_dict = {}  # selected_group_structure : test_r2

# for bt_seed in bootstrap_seeds:

print "####################"
print "# Bootstrap seed:", bt_seed
print "####################"
# Re-sampling and prepare bootstrap train data
np.random.seed(bt_seed)
sampled_idx = np.random.choice(range(ntrain), ntrain)
x = traindata.X[sampled_idx, :]
y = traindata.y[sampled_idx]
traindata_bootstrap = Data(y, x)
train_g_bootstrap = train_g[sampled_idx]

# Exhaustively train OKGT for all group structures
# on the bootstrap sample.
# The R2's are not penalize
print("=== Train all group structures without penalty ===")
trainedGroupStructures_dict = {}
for g in allGroupStructures:
    okgt = OKGTReg2(traindata_bootstrap, kernel=xKernel, group=g)
    fit = okgt._train_Vanilla2(train_g_bootstrap)
    # fit = okgt._train_lr(train_g_bootstrap)
    trainedGroupStructures_dict[g.__str__()] = fit['r2']
    print g, ' : ', fit['r2']
# Use one (mu, alpha) at a time, penalize the R2's above
# to obtain the penalized R2. We choose the group structure
# with the highest penalized R2. So each (mu, alpha) is
# corresponding to one selected group structure.
print("=== Select a group structure for each (mu, alpha) pair ===")
selectedGroupStructuresForEachParameter_dict = {}
for mu, alpha in itertools.product(muList, alphaList):
    print "mu = ", mu, ", alpha = ", alpha
    penalizedGroupStructures_dict = {}
    for g in allGroupStructures:
        penalty = mu * rkhsCapacity(g, alpha)
        penalizedGroupStructures_dict[g.__str__()] = trainedGroupStructures_dict[g.__str__()] - penalty
    selectedGroupStructure = max(penalizedGroupStructures_dict, key=penalizedGroupStructures_dict.get)
    print "selected group structure: ", selectedGroupStructure
    selectedGroupStructuresForEachParameter_dict[(mu, alpha)] = selectedGroupStructure
## It is possible that multiple (mu, alpha) pairs share the
## same group structure. So we compile a list of unique group
## structures that are selected from the training phase.
selectedGroupStructures_set = set(selectedGroupStructuresForEachParameter_dict.values())

# Use each selected group structure to fit the test set
# and keep the one with the highest R2 as the final output
print("=== Fit OKGT for the selected group structures on the test data without penalty ===")
for g in selectedGroupStructures_set:
    # g is newly found, and not been tested yet
    if g not in testedGroupStructures_dict.keys():
        okgt = OKGTReg2(testdata, kernel=xKernel, group=Group(group_struct_string=g))
        fit = okgt._train_Vanilla2(test_g)
        # fit = okgt._train_lr(test_g)
        testedGroupStructures_dict[g.__str__()] = fit['r2']
        print "group structure: ", g, ", test r2 = ", fit['r2']

# for k, v in selectedGroupStructuresForEachParameter_dict.iteritems():
#     bootstrap_dict[k].append(testedGroupStructures_dict[v])

for k, v in selectedGroupStructuresForEachParameter_dict.iteritems():
    bootstrap_dict[k] = testedGroupStructures_dict[v]

# Compile bootstrap results
# bootstrap_r2_dict = {k: np.mean(v) - np.std(v) * 0.1 for k, v in bootstrap_dict.items()}
# bestParameters = max(bootstrap_r2_dict, key = bootstrap_r2_dict.get)

# dumpObject = dict(bootstrap_r2list = bootstrap_dict,
#                   bootstrap_r2 = bootstrap_r2_dict,
#                   test = testedGroupStructures_dict,
#                   bestParameters = bestParameters)

dumpObject = dict(bootstrap_r2=bootstrap_dict,
                  test=testedGroupStructures_dict)

# Pickle the results
filename, file_extension = os.path.splitext(__file__)
filename = filename + \
           "-model" + str(model_id) + \
           "-dataseed" + str(data_seed) + \
           "-btseed" + str(bt_seed) + \
           "-" + timestamp + ".pkl"
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)
