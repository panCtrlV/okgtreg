#! /usr/bin/env python

__author__ = 'panc'

import sys, os
import numpy as np
import itertools
import pickle
from collections import defaultdict
import time

from okgtreg.simulation.sim_02052016.model import selectModel
from okgtreg.Kernel import Kernel
from okgtreg.groupStructureDetection.backwardPartition import backwardPartition
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.simulation.sim_02052016.helper import currentTimestamp

'''
Cross validation (10-fold) for choosing optimal penalty parameters.

Each data set is split into 10 patches of same size. Then run penalized
OKGT (with one (mu, alpha) pair) on 9 patches to select a model, then
use the model to fit OKGT on the remaining patch to estimate the R2. After
iterating over all 10 patches, the R2's are averaged.

The above procedure is repeated for each pair of (mu, alpha) in a
given grid. Then the best pair of parameters is determined by that
gives the largest "average R2".

#
# Pickled Objects
#
- `cv_dict`: { (mu, alpha) : { fold_num : {group_str, train_r2, test_r2} } }

- `report_dict`: {(mu, alpha) : average_of_test_R2s}

#
# Server Deployment
#


'''
start_time = time.time()

# When the script start execution YYYYMMDDHHMM
timestamp = currentTimestamp()

# Parse command line inputs: model id and tuning parameter a
args = sys.argv
model_id = int(args[1])  # model id
seed_num = int(args[2])  # seed number

# model_id = 1
model = selectModel(model_id)  # Selected model

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Simulation size (repeat for this many data sets)
# nSim = 100

# Parameter grid
muList = np.exp(np.linspace(np.log(1e-10), np.log(1. / 64), 2))  # 5
alphaList = np.arange(1, 2)  # 11

# simulate data
n = 550
np.random.seed(seed_num)
data, tgroup, g = model(n)  # data set, true group structure, true g(y)

k = 10  # k-fold
batchsize = int(np.ceil(n * 1.0 / k))

# Containers to collect results
cv_dict = defaultdict(dict)  # intermediate results
report_dict = {}  # object to pickle
for mu, alpha in itertools.product(muList, alphaList):
    # Each (mu, alpha) is evaluated by 10-fold CV
    testR2s = []
    for i in range(k):
        start = i * batchsize
        end = min((i + 1) * batchsize, n)
        # Prepare data for i-th fold
        test_data = data[start:end]
        train_data = data[0:start] + data[end:n]
        # Select model from train data
        train_res = backwardPartition(train_data, kernel, mu=mu, alpha=alpha)
        group = train_res['group']
        group_string = group.__str__()
        train_r2 = train_res['r2']
        # Fit OKGT on test data
        test_okgt = OKGTReg2(test_data, kernel=kernel, group=group)
        test_res = test_okgt.train()
        test_r2 = test_res['r2']
        testR2s.append(test_r2)
        cv_dict[(mu, alpha)][i] = dict(group=group_string, train_r2=train_r2, test_r2=test_r2)

    report_dict[(mu, alpha)] = np.mean(testR2s)

# The following is the object to be pickled
dumpObject = dict(cv_dict=cv_dict,
                  report_dict=report_dict)

# Pickle results
# dirpath = os.getcwd()
filename, file_extension = os.path.splitext(__file__)
filename = filename + \
           "-model" + str(model_id) + \
           "-seed" + str(seed_num) + \
           "-" + timestamp + ".pkl"
# saveto = dirpath + '/' + filename
saveto = filename
with open(saveto, 'wb') as f:
    pickle.dump(dumpObject, f)

time_used = time.time() - start_time
print("\n\n\nTotal time used: %.05f s. " % time_used)

# --------------------------------------------------
# # Unpickle result
# with open("okgtreg/simulation/sim_02052016/cvalidate-model1-seed-25-201602071244.pkl", 'rb') as f:
#     res = pickle.load(f)
#
# res.keys()
# k = res['cv_dict'].keys()
# res['cv_dict'][k[0]]
# res['report_dict']
