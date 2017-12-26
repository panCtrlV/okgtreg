# -*- coding: utf-8 -*-
# @Author: Pan Chao
# @Date:   2017-05-12 23:57:19
# @Last Modified by:   Pan Chao
# @Last Modified time: 2017-12-26 10:40:10

import numpy as np

from okgtreg.simulation_study.sandbox.models import selectModel
from okgtreg.Kernel import Kernel
from okgtreg.gasi.backwardPartition import backwardPartition
from okgtreg.OKGTReg import OKGTReg
from okgtreg.gasi.utility import rkhsCapacity


# Choose a model for simulation
model_id = 1
model = selectModel(model_id)

# Choose a kernel (for all groups)
kernel = Kernel('gaussian', sigma=0.5)

# Simulation
nSample = 500
## simulate data
np.random.seed(25)
data, true_group = model(nSample)  # dataset and true group structure

# Fit linear regression and calculate R^2
beta = np.linalg.inv(data.X.T.dot(data.X)).dot(data.X.T.dot(data.y))
y_hat = data.X.dot(beta)
SSE = np.sum((data.y - y_hat) ** 2)
SST = np.sum((data.y - np.mean(data.y)) ** 2)
print("R2 from fitting a liner regression: {}".format(1 - SSE / SST))

# Apply OKGT backward selection (with complexity penalty)
# backwardPartition(data, kernel, lmbda=0.1)
res = backwardPartition(data, kernel, seed=123, mu=0.01)
print("Estimated group structure: {}".format(res['group'])),
print("with R^2 = {}".format(res['r2']))

# Apply the true group structure and calculate R^2
okgt_true = OKGTReg(data, kernel=kernel, group=true_group)
fit_true = okgt_true.train()
print("R^2 when using the true group structure: {}".format(fit_true['r2']))

# Calculate RKHS complexity penalty
capacity_true = rkhsCapacity(true_group, np.e)  # 
mu = 0.01
print("Penalized R2 = {}".format(fit_true['r2'] - mu * capacity_true))
