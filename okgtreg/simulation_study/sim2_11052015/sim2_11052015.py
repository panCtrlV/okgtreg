from okgtreg.DataSimulator import *
from okgtreg.groupStructureDetection.forwardSelection import *

"""
Another example of OKGT group structure determination by
steop-wise forward selection.

In this example, we choose a model that has one interaction term,
i.e. no interactions. The model is:

    y=log(4 + sin(4 * X1) + |X2| + X3^2 + X4^3 + X5 + X6*X7 0.1*\epsilon)
    Xi ~ Unif(-1, 1)
    \epsilon ~ N(0, 1)
"""

# Simulation data
np.random.seed(25)
y, X = DataSimulator.SimData_Wang04WithInteraction(1000)  # Simulate data
data = Data(y, X)

# Same kernel for all groups
kernel = Kernel('gaussian', sigma=0.5)

# Step-wise foward selection to determine group structure
res = forwardSelection(data, kernel, useLowRankApproximation=True, rank=10)

"""Simulation results:

In this example, we successfully recover the true group structure
including the interaction:

    ** SELECTED GROUP STRUCTURE: ([1], [2], [3], [4], [5], [6, 7])
"""


