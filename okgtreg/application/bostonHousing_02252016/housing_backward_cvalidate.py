__author__ = 'panc'

'''
Backward selection with 10-fold cross validation.

Suitable when the sample size is limited.
'''

import sys, os
import numpy as np
import pickle

from okgtreg.Kernel import Kernel
from okgtreg.utility import currentTimestamp
from okgtreg.DataUtils import readHousingData
from okgtreg.groupStructureDetection.backwardPartition import backwardPartitionWithKnownResponse
from okgtreg.application.bostonHousing_02252016.utility import predict, predictionError

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

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Read data
data = readHousingData()

# Split data into 10 folds
k = 10  # 10-folds
batchsize = int(np.ceil(data.n * 1.0 / k))

## Given (mu, alpha)
train_gstructs_dict = {}
train_r2_dict = {}
train_fcall_dict = {}
test_errors_dict = {}
for i in range(k):
    start = i * batchsize
    end = min((i + 1) * batchsize, data.n)
    # the last batch will be smaller if data.n
    #   is no k-divisible
    # print start, end
    # Prepare data for i-th fold
    test_data = data[start:end]
    train_data = data[0:start] + data[end:data.n]
    # Group selection and model estimation from train data
    train_res = backwardPartitionWithKnownResponse(train_data, kernel, train_data.y,
                                                   mu=mu, alpha=alpha)
    selected_gstruct = train_res['group']
    train_gstructs_dict[i + 1] = selected_gstruct
    train_r2_dict[i + 1] = train_res['r2']  # penalized R2
    estimated_fns = train_res['f_call']
    train_fcall_dict[i + 1] = estimated_fns
    print str(i + 1) + ':' + train_res['group'].__str__()
    print "\tR2 (penalized) = %.10f" % train_res['r2']
    # Calculate test/prediction error
    test_yhat = predict(test_data, kernel, selected_gstruct, estimated_fns)
    test_error = predictionError(test_data.y, test_yhat)
    test_errors_dict[i + 1] = test_error
    print "\tTest error = %.10f" % test_error

avg_test_error = np.mean(test_errors_dict.values())
print "Average test error = %.10f" % avg_test_error

# Pickle / Save results
dumpObject = dict(train_gstructs=train_gstructs_dict,
                  train_fcalls=train_fcall_dict,
                  train_r2=train_r2_dict,
                  test_errors=test_errors_dict,
                  avg_test_error=avg_test_error)

filename, file_extension = os.path.splitext(__file__)
filename = filename + "-" + \
           "mu" + str(mu_id) + "-" + \
           "alpha" + str(alpha_id) + "-" + \
           timestamp + ".pkl"
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)
