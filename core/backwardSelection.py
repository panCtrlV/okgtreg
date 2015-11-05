import numpy as np

from core.Group import *
from core.okgtreg import *
from core.DataSimulator import *


"""
Determining group structure by backward selection.
We start from a fully non-parametric model, i.e.

    g(y) = f(x_1, x_2, ..., x_p)

and record its r2.

Then for each one in the pool of available variables,
we try the following operations:

1. Create a new group where the selected variable is
   the only memeber. Then, okgt is fitted using the new
   group structure.

2. If there are other groups than the pool group we
   started from, the selected variable joins each
   of the other groups to form a new group structure.
   Then, okgt is fitted using the new group structure.

During each operation, the new r2 is compared with the
old r2 to check if there is improvement. If r2 increases,
r2 is updated.

The above procedure continues until one of the following
conditions is met:

1. r2 stops increasing
2. there is only one variable remaining in the pool

The second condition is set to avoid redundancy. For example,
if there are two variable in total (x1, x2). The possible
group structures are either (x1, x2) or [(x1), (x2)]. That is,
once a variable is separated from the other, the procedure
is done. There is no need to test for the last variable in the
pool.
"""

# Simulate data
y, x = DataSimulator.SimData_Wang04(500)
data = Data(y, x)


proceed = True  # flag if the algorithm continues
covariatesPool = list(np.arange(data.p) + 1)
oldGroup = Group(covariatesPool)

while proceed:
    for covariateInd in covariatesPool:
        currentGroup = old.remove