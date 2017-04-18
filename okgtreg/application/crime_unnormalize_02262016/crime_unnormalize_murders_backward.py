__author__ = 'panc'

'''
Apply backward method to the data for Murders
'''
import sys, os
import numpy as np
import pickle

from okgtreg.utility import currentTimestamp
from okgtreg.application.crime_unnormalize_02262016.utility import readCleanDataForMurders
from okgtreg.Data import Data
from okgtreg.Kernel import Kernel
from okgtreg.groupStructureDetection.backwardPartition import backwardPartitionWithKnownResponse
from okgtreg.application.crime_unnormalize_02262016.utility import predict, predictionError

# Current time
timestamp = currentTimestamp()

# Parse command line arguments
args = sys.argv
mu_id = int(args[1])  # mu_id 1~5
alpha_id = int(args[2])  # alpha_id 1~10

# Parameter grid
mu_size = 5
alpha_size = 10
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
alphaList = np.arange(1, alpha_size + 1)

mu = muList[mu_id - 1]
alpha = alphaList[alpha_id - 1]
print "mu =", mu, ", alpha =", alpha

################
# Prepare Data #
################
# Read data
data = readCleanDataForMurders()

# Prepare training and testing data sets
## Randomly split data to train and test sets
np.random.seed(25)
## test set (100)
ntest = 100
ntrain = data.n - ntest
test_idx = np.random.choice(range(data.n), size=ntest, replace=False)
test_data = Data(y=data.y[test_idx], X=data.X[test_idx, :])
## train set
train_idx = [x for x in range(data.n) if x not in test_idx]
train_data = Data(y=data.y[train_idx], X=data.X[train_idx, :])

##########
# Kernel #
##########
kernel = Kernel('gaussian', sigma=0.5)

########################################################
# For each (mu, alpha), run backward selection on the  #
#   training data, then calculate the prediction error #
#   on the validation data                             #
########################################################
train_dict = {}
test_dict = {}

backward_res = backwardPartitionWithKnownResponse(train_data, kernel, train_data.y,
                                                  mu=mu, alpha=alpha)
train_dict[(mu, alpha)] = backward_res  # save result
print "Selected group structure: ", backward_res['group']
yhat = predict(test_data, kernel, backward_res['group'], backward_res['f_call'])
test_error = predictionError(test_data.y, yhat)
test_dict[(mu, alpha)] = test_error
print "Test error: %.10f" % test_error

###########################
# Pickle/Save the results #
###########################
dumpObject = dict(train=train_dict,
                  test=test_dict)

filename, file_extension = os.path.splitext(__file__)
filename = filename + "_" + \
           "mu" + str(mu_id) + "-" + \
           "alpha" + str(alpha_id) + "-" + \
           timestamp + ".pkl"
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)
