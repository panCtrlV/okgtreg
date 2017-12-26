__author__ = 'panc'

from okgtreg.simulation.sim_02052016.model import selectModel
from okgtreg.Kernel import Kernel
from okgtreg.groupStructureDetection.backwardPartition import backwardPartitionWithKnownResponse
from okgtreg.simulation.sim_02052016.helper import currentTimestamp
from okgtreg.Group import Group
from okgtreg.simulation.sim_02052016.helper import predict, predictionError

import numpy as np
import sys, os
import pickle
import itertools
from collections import defaultdict

# Parse command line inputs: model id and tuning parameter a
args = sys.argv
model_id = int(args[1])  # model id
seed_num = int(args[2])  # seed for data simulation
# Selected model
model = selectModel(model_id)  # Selected model

# Current time
timestamp = currentTimestamp()

# Kernel:
# The response transformation is assumed to be known, so
# we only need to transform each group of the covariates.
xKernel = Kernel('gaussian', sigma=0.5)

# Grid of values for tuning parameters (mu, alpha)
#   For each data set, apply one-fold validation for all value
#   pairs (50) in the parameter grid for each possible group
#   structures
mu_size = 5
alpha_size = 10
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
alphaList = np.arange(1, alpha_size + 1)

# Print INFO
print "###############################################"
print "# Exhaustive Backward Selection"
print "# Model %d" % model_id
print "# Seed = %d" % seed_num
print "# Kernel: Gaussian (0.5). Response fixed."
print "# Mu values: ", muList
print "# Alpha values: ", alphaList
print "# OKGT fitting method: LSR with intercept"
print "###############################################"

# Prepare data sets (train + test)
ntrain = 500  # train size
ntest = 500  # test size

np.random.seed(seed_num)
train_data, true_train_group, train_g = model(ntrain)
test_data, true_test_group, test_g = model(ntest)

# For each (mu, alpha) use backward method to select
#   a group structure.
#   For each OKGT fitting, estimate the transformation
#   function and the R2 (without penalty).
print("################################################")
print("# Backward selection for each (mu, alpha) pair")
print("################################################")
train_r2_dict = {} # save penalized R2
# train_f_dict = {}
train_gstruct_dict = {} # save group structures
gstruct_f_dict = {} # save transformation functions
for mu, alpha in itertools.product(muList, alphaList):
    print "mu =", mu, ", alpha =", alpha
    train_res = backwardPartitionWithKnownResponse(train_data, xKernel, train_data.y,
                                                   mu=mu, alpha=alpha)
    train_gstruct_str = train_res['group'].__str__()
    train_r2_dict[(mu, alpha)] = train_res['r2']
    # train_f_dict[(mu,alpha)] = train_res['f_call']
    if gstruct_f_dict.get(train_gstruct_str) is None:
        gstruct_f_dict[train_gstruct_str] = train_res['f_call']
    train_gstruct_dict[(mu, alpha)] = train_gstruct_str
    print " => group structure:", train_res['group'].__str__()
    print " => R2 (penalized) =", train_res['r2']

# Different (mu, alpha) pairs might choose the same
#   group structure. So we collect the UNIQUE group
#   structures selected from the training phase.
uniqueSelectedGroupStructures_set = set(train_gstruct_dict.values())

# Use estimated transformation functions for each (mu, alpha) pair
#   (corresponding to the selected group structure to calculate the
#   prediction error for the test set.
# Then keep the pair with the highest R2 as the final selected tuning
#   parameters.
print("#############################################################################")
print("# Calculate Prediction Error for Each Selected Group Structure on Test Data")
print("#############################################################################")
testedGroupStructures_error_dict = defaultdict(dict)
for gstruct_str in uniqueSelectedGroupStructures_set:
    cur_fns = gstruct_f_dict[gstruct_str]
    cur_gstruct = Group(group_struct_string=gstruct_str)
    test_yhat = predict(test_data, xKernel, cur_gstruct, cur_fns)
    test_error = predictionError(test_data.y, test_yhat)
    testedGroupStructures_error_dict[gstruct_str] = test_error
    print "group structure {0:30} : test error = {1:10}".format(gstruct_str, "%.10f" % test_error)

print("######################")
print("# One-fold CV Result")
print("######################")
bestGroupStructure_str = min(testedGroupStructures_error_dict,
                             key=testedGroupStructures_error_dict.get)
print "Best group structure: ", bestGroupStructure_str
print "Best tuning parameters: "
bestTuningParameters = []
for k, v in train_gstruct_dict.iteritems():
    if v == bestGroupStructure_str:
        print k
        bestTuningParameters.append(k)

###########################
# Pickle/Save the results #
###########################

# Object to pickle
dumpObject = dict(train_r2=train_r2_dict,
                  gstruct_f=gstruct_f_dict,
                  select=train_gstruct_dict,  # selected group structure from training for each (mu, alpha)
                  test_error=testedGroupStructures_error_dict,
                  bestGroupStructure=bestGroupStructure_str,
                  bestTuningParameters=bestTuningParameters)

# Pickle to file
filename, file_extension = os.path.splitext(__file__)
filename = filename + \
           "-model" + str(model_id) + \
           "-seed" + str(seed_num) + \
           "-" + timestamp + ".pkl"
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)
