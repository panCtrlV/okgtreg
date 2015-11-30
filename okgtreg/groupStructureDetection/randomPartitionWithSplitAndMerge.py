"""
We start with a random partition of the predictor variables. The corresponding
OKGT is fitted with R2 being recorded. Then perform the following split and join
operators:

1. For each group of size > 1, split it into individual covariates and fit
   the corresponding OKGT and record its R2.

   Compare R2 under each scenario in (2) with the R2 we started with. Pick the
   the scenario with gives the largest improvement in R2 and randomly pick a
   covariate from the group to form a uni-variate group.

2. For each univariate group in the current structure, try to merge its covariate
   with one of the other groups, regardless of the other group being univariate or
   multivariate.

Then, step 1 and 2 are repeated iteratively until no further improvement in R2.

Can be extended to include removing steps for variable selection.

Question:

1. If different random partitions, do they result in the same optimal group structure?
"""

import numpy as np

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Group import Group, RandomGroup
from okgtreg.OKGTReg import OKGTReg
from okgtreg.Kernel import Kernel
from okgtreg.Parameters import Parameters

kernel = Kernel('gaussian', sigma=0.5)

# Simulate data
np.random.seed(25)
data = DataSimulator.SimData_Wang04WithInteraction2(100)

# True group
trueGroup = Group([1], [2], [3], [4], [5], [6,7,8])
trueParameter = Parameters(trueGroup, kernel, [kernel]*trueGroup.size)
trueOkgt = OKGTReg(data, trueParameter)
res = trueOkgt.train()
res['r2']

# Random partition to start with, where
# the number of groups are pre-determined.
group0 = RandomGroup(4, [i+1 for i in range(data.p)])
parameters0 = Parameters(group0, kernel, [kernel]*group0.size)
okgt0 = OKGTReg(data, parameters0)

# # Detecting group structure by split and merge
# counter = 0
# okgt_beforeSplit = okgt0
#
# while counter < 4:
#     print("=== counter: %d ====" % counter)
#     okgt_afterSplit = okgt_beforeSplit.optimalSplit(kernel)
#     okgt_afterMerge = okgt_afterSplit.optimalMerge(kernel)
#     res = okgt_afterMerge.train()
#     print res['r2']
#     okgt_beforeSplit = okgt_afterMerge
#     counter+=1

# --- split ---
okgt_afterSplit = okgt0.optimalSplit(kernel, method='vanilla')
