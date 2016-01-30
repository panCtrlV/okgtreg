__author__ = 'panc'

from okgtreg.simulation.sandbox.models import *
from okgtreg.groupStructureDetection.backwardPartition import *

# Selected model
model_id = 1
model = selectModel(model_id)

# Kernel
kernel = Kernel('gaussian', sigma=0.5)

# Simulation
nSample = 500

## simulate data
np.random.seed(25)
data, tgroup = model(nSample)

# linear regression
beta = np.linalg.inv(data.X.T.dot(data.X)).dot(data.X.T.dot(data.y))
y_hat = data.X.dot(beta)
SSE = np.sum((data.y - y_hat) ** 2)
SSE
SST = np.sum((data.y - np.mean(data.y)) ** 2)
SST
1 - SSE / SST

# backward selection
backwardPartition(data, kernel, lmbda=0.1)

# using true group structure
okgt_true = OKGTReg(data, kernel=kernel, group=tgroup)
fit_true = okgt_true.train()
# tgroup
print fit_true['r2']

capacity_true = rkhsCapacity(tgroup, np.e)
lmbda = 1e-5
print fit_true['r2'] - lmbda * capacity_true
