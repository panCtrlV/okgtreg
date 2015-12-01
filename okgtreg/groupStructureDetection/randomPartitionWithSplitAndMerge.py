"""
This simulation illustrate step-by-step the procedure of group structure determination
proposed by Michael.

We start with a random partition of the predictor variables. The corresponding
OKGT is fitted with R2 being recorded. Then we perform the following split and
merge operators until there is no further improvement in R2:

1. For each group of size > 1, split it into individual covariates and fit
   the resulting OKGT and record its R2.

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
data = DataSimulator.SimData_Wang04WithInteraction2(500)

# True group
trueGroup = Group([1], [2], [3], [4], [5], [6,7,8])
trueParameter = Parameters(trueGroup, kernel, [kernel]*trueGroup.size)
trueOkgt = OKGTReg(data, trueParameter)
res = trueOkgt.train('nystroem', 10)
res['r2']

# Random partition to start with, where
# the number of groups are pre-determined.
np.random.seed(25)
group0 = RandomGroup(4, [i+1 for i in range(data.p)])
parameters0 = Parameters(group0, kernel, [kernel]*group0.size)
okgt0 = OKGTReg(data, parameters0)
okgt0.getGroupStructure()


################################
# Split and merge step-by-step #
################################
# Nystroem is a random projection method, so without fixing seed,
# the model fitting (i.e. R2) is different each time.
seed = 25
# --- 1 ---
# if split
okgt_afterSplit = okgt0.optimalSplit(kernel, method='nystroem', nComponents=10, seed=seed)
# if merge
okgt_afterMerge = okgt0.optimalMerge(kernel, method='nystroem', nComponents=10, seed=seed)

# >>> split is more optimal
okgt1 = okgt_afterSplit

# --- 2 ---
# if split
okgt_afterSplit = okgt1.optimalSplit(kernel, method='nystroem', nComponents=10, seed=seed)
# if merge
okgt_afterMerge = okgt1.optimalMerge(kernel, method='nystroem', nComponents=10, seed=seed)

# >>> split is more optimal
okgt2 = okgt_afterSplit

# --- 3 ---
okgt_afterMerge = okgt2.optimalMerge(kernel, method='nystroem', nComponents=10, seed=seed)

# >>> merge is the only option, and R2 improves
okgt3 = okgt_afterMerge

# --- 4 ---
# if split
okgt_afterSplit = okgt3.optimalSplit(kernel, method='nystroem', nComponents=10, seed=seed)
# if merge
okgt_afterMerge = okgt3.optimalMerge(kernel, method='nystroem', nComponents=10, seed=seed)

# >>> merge is more optimal
okgt4 = okgt_afterMerge

# --- 5 ---
# if split
okgt_afterSplit = okgt4.optimalSplit(kernel, method='nystroem', nComponents=10, seed=seed)
# if merge
okgt_afterMerge = okgt4.optimalMerge(kernel, method='nystroem', nComponents=10, seed=seed)

# >>> no change after split and merge, stop.


##############
# While loop #
##############