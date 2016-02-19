__author__ = 'panc'

'''
1-fold validation for group structure selection and choosing
 tuning parameters.

The training phase uses the exhaustive search (instead of a
stepwise method). So the procedure contains the following steps:

1. For each possible group structure, fit the OKGT to estimate R2
2. Adjust each R2 with different pair of (mu, alpha) to obtained the
   adjust R2.
   (Note: after step 2, we end up with a (# group structures)*(# parameter pairs) matrix)
3. For each parameter pair, select the group structure that possesses
   the largest adjusted R2
4. Use the group structure selected from each (mu, alpha) pair to
   fit an OKGT on the test data.
5. We selected the group structure that results in the largest R2
   from test data. The corresponding (mu, alpha) pair(s) is(are)
   considered as optimal.
'''

from okgtreg.simulation.sim_02052016.model import selectModel
from okgtreg.Kernel import Kernel
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.groupStructureDetection.backwardPartition import rkhsCapacity
from okgtreg.simulation.sim_02052016.helper import currentTimestamp
from okgtreg.utility import partitions
from okgtreg.Group import Group

import numpy as np
import sys, os
import pickle
import itertools
from collections import defaultdict

# model_id = 1
# seed_num = 25
mu_size = 5
alpha_size = 10

# Parse command line inputs: model id and tuning parameter a
args = sys.argv
model_id = int(args[1])  # model id
seed_num = int(args[2])  # seed for data simulation

# current time
timestamp = currentTimestamp()

# selected model
model = selectModel(model_id)  # Selected model

# Kernel:
# The response transformation is assumed to be known,
# so we only need to transform the covariates.
xKernel = Kernel('gaussian', sigma=0.5)

# For each data set, apply one-fold validation for all value
# pairs (50) in the parameter grid for each possible group
# structures (203 different structures for 6 covariates)
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
alphaList = np.arange(1, alpha_size + 1)

allPartitions = [tuple(list(item) for item in p) for p in partitions(set(np.arange(6) + 1))]
allGroupStructures = [Group(*p) for p in allPartitions]

# Print INFO
print "###############################################"
print "# Model %d" % model_id
print "# Seed = %d" % seed_num
print "# Kernel: Gaussian (0.5). Response fixed."
print "# Mu values: ", muList
print "# Alpha values: ", alphaList
print "###############################################"

# Simulate a data set (train + test)
ntrain = 500
ntest = 200

np.random.seed(seed_num)
traindata, tgroup, train_g = model(ntrain)
testdata, testgroup, test_g = model(ntest)

# Exhaustively train OKGT for all group structures,
# the R2's are not penalized
print("#############################################")
print("# Train all group structures without penalty ")
print("#############################################")
trainedGroupStructures_dict = defaultdict(dict)
for g in allGroupStructures:
    okgt = OKGTReg2(traindata, kernel=xKernel, group=g)
    fit = okgt._train_Vanilla2(train_g)
    trainedGroupStructures_dict[g.__str__()] = fit['r2']
    print g, ' : ', fit['r2']

# Use one (mu, alpha) pair at a time, penalize the R2's above
# to obtain the penalized R2. We choose the group structure
# with the highest penalized R2. So each (mu, alpha) is
# corresponding to one selected group structure.
print("#######################################################")
print("# Select a group structure for each (mu, alpha) pair ")
print("#######################################################")
selectedGroupStructuresForEachParameter_dict = defaultdict(dict)
for mu, alpha in itertools.product(muList, alphaList):
    print "mu = ", mu, ", alpha = ", alpha
    penalizedGroupStructures_dict = defaultdict(dict)
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
print("#############################################################################")
print("# Fit OKGT for the selected group structures on the test data without penalty")
print("#############################################################################")
testedGroupStructures_dict = defaultdict(dict)
for g in selectedGroupStructures_set:
    okgt = OKGTReg2(testdata, kernel=xKernel, group=Group(group_struct_string=g))
    fit = okgt._train_Vanilla2(test_g)
    testedGroupStructures_dict[g.__str__()] = fit['r2']
    print "group structure: ", g, ", test r2 = ", fit['r2']

bestGroupStructure = max(testedGroupStructures_dict, key=testedGroupStructures_dict.get)
print "Best group structure: ", bestGroupStructure
print "Best tuning parameters: "
bestTuningParameters = []
for k, v in selectedGroupStructuresForEachParameter_dict.iteritems():
    if v == bestGroupStructure:
        print k
        bestTuningParameters.append(k)

dumpObject = dict(train=trainedGroupStructures_dict,
                  select=selectedGroupStructuresForEachParameter_dict,
                  test=testedGroupStructures_dict,
                  bestGroupStructure=bestGroupStructure,
                  bestTuningParameters=bestTuningParameters)

# Pickle the results
filename, file_extension = os.path.splitext(__file__)
filename = filename + \
           "-model" + str(model_id) + \
           "-seed" + str(seed_num) + \
           "-" + timestamp + ".pkl"
# saveto = dirpath + '/' + filename
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)
