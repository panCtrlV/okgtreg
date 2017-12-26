__author__ = 'panc'

'''
1-fold validation for group structure selection and choosing
 tuning parameters.

The training phase uses the exhaustive search (instead of a
stepwise method). So the procedure contains the following steps:

1. For each possible group structure, fit the OKGT to estimate R2
   and the transformation functions f1, ..., fd (suppose there are
   d groups in the group structure).

Note: Fitting OKGT is done through linear regression where the gram
      matrix of the additive kernel is used as the design matrix,
      the response is the true transformation.

2. Adjust each R2 with different pair of (mu, alpha) to obtained the
   adjust R2.

Note: After step 2, we end up with a (# group structures)*(# parameter pairs) matrix)

3. For each parameter pair, retain the group structure that possesses
   the largest adjusted R2 as the optimal group structure from the
   training phase. Also, keep the corresponding estimator of the
   transformation functions.

4. Use the estimated transformation functions estimated from each
   (mu, alpha) pair to calculate the prediction error (quadratic loss)
   on the test set.

5. The tuning parameters (mu, alpha) that results in the smallest
   prediction error is considered to be optimal.
'''

from okgtreg.simulation.sim_02052016.model import selectModel
from okgtreg.Kernel import Kernel
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.groupStructureDetection.backwardPartition import rkhsCapacity
from okgtreg.simulation.sim_02052016.helper import currentTimestamp
from okgtreg.utility import partitions
from okgtreg.Group import Group
from okgtreg.Parameters import Parameters
from okgtreg.Data import ParameterizedData

import numpy as np
import sys, os
import pickle
import itertools
from collections import defaultdict

# model_id = 4
# seed_num = 1


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
# All possible group structures
#   (203 different structures for 6 covariates)
allPartitions = [tuple(list(item) for item in p) for p in partitions(set(np.arange(6) + 1))]
allGroupStructures = [Group(*p) for p in allPartitions]

# |
# | Print INFO
print "###############################################"
print "# Model %d" % model_id
print "# Seed = %d" % seed_num
print "# Kernel: Gaussian (0.5). Response fixed."
print "# Mu values: ", muList
print "# Alpha values: ", alphaList
print "# OKGT fitting method: LSR"
print "###############################################"
# |
# | Prepare data sets (train + test)
ntrain = 500  # train size
ntest = 500  # test size

np.random.seed(seed_num)
train_data, true_train_group, train_g = model(ntrain)
test_data, true_test_group, test_g = model(ntest)
# |
# | Exhaustively train OKGT for all group structures.
# |  For each OKGT fitting, estimate the transformation
# |  function and the R2 (without penalty).
print("#############################################")
print("# Train All Group Structures Without Penalty ")
print("#############################################")
trainedGroupStructures_r2_dict = defaultdict(dict)
trainedGroupStructures_f_dict = defaultdict(dict)
for gstruct in allGroupStructures:
    okgt = OKGTReg2(train_data, kernel=xKernel, group=gstruct)
    # fit = okgt._train_Vanilla2(train_g)
    fit = okgt._train_lr(train_g)
    # Record R2 (no penalty) for each group structure
    #   group_struct_str : train_r2_no_penalty
    trainedGroupStructures_r2_dict[gstruct.__str__()] = fit['r2']
    # Record estimated tranformation functions for each group structure
    #   group_struct_str : {group_number : callable}
    trainedGroupStructures_f_dict[gstruct.__str__()] = fit['f_call']
    print gstruct.__str__(), ' : ', fit['r2']

# Use one (mu, alpha) pair at a time, penalize the R2's above
#   to obtain the penalized R2. We choose the group structure
#   with the highest penalized R2. So each (mu, alpha) is
#   corresponding to one selected group structure.
print("#############################################################")
print("# Select the Best Group Structure for Each (mu, alpha) Pair ")
print("#############################################################")
selectedGroupStructuresForEachParameter_dict = defaultdict(dict)
for mu, alpha in itertools.product(muList, alphaList):
    print "mu = {0:10}, alpha = {1:5}".format(mu, alpha)
    penalizedGroupStructures_r2_dict = defaultdict(dict)
    for gstruct in allGroupStructures:
        # calculate the penalty
        penalty = mu * rkhsCapacity(gstruct, alpha)
        # calculate the penalized R2
        penalizedGroupStructures_r2_dict[gstruct.__str__()] = \
            trainedGroupStructures_r2_dict[gstruct.__str__()] - penalty
    selectedGroupStructure_str = max(penalizedGroupStructures_r2_dict,
                                     key=penalizedGroupStructures_r2_dict.get)
    print "\tselected group structure: ", selectedGroupStructure_str
    # Record the selected group structure for (mu, alpha)
    #   (mu, alpha) : group_struct_str
    selectedGroupStructuresForEachParameter_dict[(mu, alpha)] = selectedGroupStructure_str
## It is possible that multiple (mu, alpha) pairs share the
## same group structure. So we compile a set of UNIQUE group
## structures that are selected from the training phase.
uniqueSelectedGroupStructures_set = set(selectedGroupStructuresForEachParameter_dict.values())

# Use estimated transformation functions for each (mu, alpha) pair
# (corresponding to the selected group structure to calculate the
# prediction error for the test set.
# Then keep the pair with the highest R2 as the final selected tuning
# parameters.
print("#############################################################################")
print("# Calculate Prediction Error for Each Selected Group Structure on Test Data")
print("#############################################################################")
# testedGroupStructures_r2_dict = defaultdict(dict)
testedGroupStructures_error_dict = defaultdict(dict)
for gstruct_str in uniqueSelectedGroupStructures_set:
    cur_gstruct = Group(group_struct_string=gstruct_str)
    # === deprecated ===
    # okgt = OKGTReg2(test_data, kernel=xKernel, group=Group(group_struct_string=gstruct_str))
    # fit = okgt._train_Vanilla2(test_g)
    # fit = okgt._train_lr(test_g)
    # testedGroupStructures_dict[g.__str__()] = fit['r2']
    # ==================
    def predict(test_data, kernel, gstruct, fns_est):
        parameters = Parameters(gstruct, kernel, [kernel] * gstruct.size)
        parameterizedTestData = ParameterizedData(test_data, parameters)
        test_g_pred_list = [fns_est[i](parameterizedTestData.getXFromGroup(i))
                            for i in range(1, gstruct.size + 1)]
        test_g_pred = sum(test_g_pred_list) + fns_est[0]  # with intercept
        return test_g_pred
    # Calculate the prediction error for the test set
    # TODO: the second xKernel is redundant for fixed response case.
    # todo: Should consider changing the interface of OKGTReg and related
    # todo: classes to incorporated fixed response more smoothly.
    cur_parameters = Parameters(cur_gstruct, xKernel, [xKernel] * cur_gstruct.size)
    cur_parameterizedTestData = ParameterizedData(test_data, cur_parameters)
    ## f estimated from training phase for this group structure
    cur_fns = trainedGroupStructures_f_dict[gstruct_str]
    ## compute the predicted value for X_test
    test_f_hat_list = [cur_fns[i](cur_parameterizedTestData.getXFromGroup(i))
                       for i in range(1, cur_gstruct.size + 1)]
    test_g_hat = sum(test_f_hat_list) + cur_fns[0]  # also add intercept
    # test_g_pred = np.column_stack(test_g_pred_list).sum(axis=1)
    test_error = sum((test_g - test_g_hat) ** 2) / ntest
    testedGroupStructures_error_dict[gstruct_str] = test_error
    print "group structure {0:30} : test error = {1:10}".format(gstruct_str, "%.10f" % test_error)
    # test_r2 = 1 - sum((test_g - test_g_pred) ** 2) / sum((test_g - np.mean(test_g)) ** 2)
    # testedGroupStructures_r2_dict[gstruct_str] = test_r2
    # print "group structure {0:30} : test R2 = {1:10}".format(gstruct_str, "%.10f" % test_r2)

print("######################")
print("# One-fold CV Result")
print("######################")
# bestGroupStructure = max(testedGroupStructures_r2_dict, key=testedGroupStructures_r2_dict.get)
bestGroupStructure = min(testedGroupStructures_error_dict,
                         key=testedGroupStructures_error_dict.get)
print "Best group structure: ", bestGroupStructure
print "Best tuning parameters: "
bestTuningParameters = []
for k, v in selectedGroupStructuresForEachParameter_dict.iteritems():
    if v == bestGroupStructure:
        print k
        bestTuningParameters.append(k)

# Object to pickle
dumpObject = dict(train_r2=trainedGroupStructures_r2_dict,
                  train_f=trainedGroupStructures_f_dict,
                  select=selectedGroupStructuresForEachParameter_dict,
                  test_error=testedGroupStructures_error_dict,
                  bestGroupStructure=bestGroupStructure,
                  bestTuningParameters=bestTuningParameters)

# Pickle/Save the results
filename, file_extension = os.path.splitext(__file__)
filename = filename + \
           "-model" + str(model_id) + \
           "-seed" + str(seed_num) + \
           "-" + timestamp + ".pkl"
# saveto = dirpath + '/' + filename
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)
