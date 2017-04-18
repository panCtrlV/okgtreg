__author__ = 'panc'

'''
Backward stepwise method of group structure identification (GSI) for
Boston Housing data set.

In order to select the best tuning parameter, a one-fold validation
is used. In particular, the data set is split into training (300) and
validation (206) sets. For each (mu, alpha) pair, the backward algorithm
is run with the corresponding penalty applied on R2. The updated group
structure is determined based on the penalized R2.
'''
import pickle
import sys, os
import itertools
import numpy as np

from okgtreg.DataUtils import readHousingData
from okgtreg.Kernel import Kernel
from okgtreg.groupStructureDetection.backwardPartition import backwardPartitionWithKnownResponse
from okgtreg.Data import Data
# from okgtreg.utility import predict, predictionError
from okgtreg.utility import currentTimestamp
from okgtreg.application.bostonHousing_02252016.utility import predict, predictionError

# Current time
timestamp = currentTimestamp()

# Parse command line arguments
args = sys.argv
mu_id = int(args[1])  # mu_id 1~5
alpha_id = int(args[2])  # alpha_id 1~10

# mu_id = 3
# alpha_id = 4

# Parameter grid
mu_size = 5
alpha_size = 10
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), mu_size))
alphaList = np.arange(1, alpha_size + 1)

mu = muList[mu_id - 1]
alpha = alphaList[alpha_id - 1]
print "mu =", mu, ", alpha =", alpha

# Read data
data = readHousingData()

# Randomly split data to train and test sets
np.random.seed(25)
## train set
ntrain = 300
train_idx = np.random.choice(range(data.n), size=ntrain, replace=False)
train_data = Data(y=data.y[train_idx], X=data.X[train_idx, :])
## test set
test_idx = [x for x in range(data.n) if x not in train_idx]
test_data = Data(y=data.y[test_idx], X=data.X[test_idx, :])

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# For each (mu, alpha), run backward selection on the
#   training data, then calculate the prediction error
#   on the validation data
train_dict = {}
test_dict = {}
# for mu, alpha in itertools.product(muList, alphaList):
#     backward_res = backwardPartitionWithKnownResponse(train_data, kernel, train_data.y,
#                                                       mu=mu, alpha=alpha)
#     train_dict[(mu, alpha)] = backward_res # save result
#     yhat = predict(test_data, kernel, backward_res['group'], backward_res['f_call'])
#     pred_error = predictionError(test_data.y, yhat)
#     test_dict[(mu, alpha)] = pred_error

backward_res = backwardPartitionWithKnownResponse(train_data, kernel, train_data.y,
                                                  mu=mu, alpha=alpha)
train_dict[(mu, alpha)] = backward_res  # save result
print "Selected group structure: ", backward_res['group']
yhat = predict(test_data, kernel, backward_res['group'], backward_res['f_call'])
pred_error = predictionError(test_data.y, yhat)
test_dict[(mu, alpha)] = pred_error
print "Test error: %.10f" % pred_error

# Pickle/Save the results
dumpObject = dict(train=train_dict,
                  test=test_dict)

filename, file_extension = os.path.splitext(__file__)
filename = filename + "-" + timestamp + ".pkl"
# saveto = dirpath + '/' + filename
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)
