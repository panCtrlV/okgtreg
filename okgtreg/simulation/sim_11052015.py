from okgtreg.forwardSelection import *
from okgtreg.DataSimulator import *

"""
An example of OKGT group structure determination by
steop-wise forward selection.

In this example, we choose a model which is fully additive,
i.e. no interactions. The model is:

    y=log(4 + sin(4 * X1) + |X2| + X3^2 + X4^3 + X5 + 0.1*\epsilon)
    Xi ~ Unif(-1, 1)
    \epsilon ~ N(0, 1)
"""

# Simulation data
np.random.seed(25)
y, X = DataSimulator.SimData_Wang04(1000)  # Simulate data
data = Data(y, X)

# Same kernel for all groups
kernel = Kernel('gaussian', sigma=0.5)

# Step-wise foward selection to determine group structure
res = forwardSelection(data, kernel, useLowRankApproximation=True, rank=10)

"""Simulation result:

In this example, we successfully recover the true group structure:

    ** SELECTED GROUP STRUCTURE: ([1], [2], [3], [4], [5])

Note: The algorithm is tested under two setting:

        1) enabling low rank approximation (using Nystroem method),
        2) no low rank approximation.

      The performance of the algorithm under the two settings are different.
      It seems that the detected structure from the first setting is closer
      to the true structure than that from the second setting.

      It was also noticed that the sample size affects the final result.

      Using low rank approximation also improve the computation speed of the
      algorithm.
"""