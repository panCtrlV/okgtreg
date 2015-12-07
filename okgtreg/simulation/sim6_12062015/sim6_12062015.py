"""
** This simulation is similar to "sim5", except that the grouped covariates
are multiplied by 100 to make it dominant in the structure. Based on the pilot
simulation, it seems that the magnitude of the groups will affect the group
structure recovery. **

This simulation test if the Split and Merger with Random Initial can recover
the true group structure.

The model used here is:

        y = np.log(4.0 +
                   np.sin(4 * x[:, 0]) +
                   np.abs(x[:, 1]) +
                   x[:, 2]**2 +
                   x[:, 3]**3 +
                   x[:, 4] +
                   100. * abs(x[:, 5] * x[:, 6] * x[:, 7]) +    <- here is the difference
                   0.1 * noise)
"""

# ---
# The following two lines added the root of the package into the system path
# so that the module okgtreg can be imported properly.
# ---
import sys
sys.path.append('../okgtreg')

# --- Simulation code starts from here ---
import numpy as np
import pickle

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Kernel import Kernel
from okgtreg.groupStructureDetection.randomPartitionWithSplitAndMerge import splitAndMergeWithRandomInitial


nSim = 100
seeds = range(nSim)

# Same data
n = 500

np.random.seed(25)
data = DataSimulator.SimData_Wang04WithInteraction2_100(500)

# Same kernel
kernel = Kernel('gaussian', sigma=0.5)

# Empty lists to save simulation results
estimatedGroupStructures = []
estimatedR2s = []

for seed in seeds:
    optimalOkgt = splitAndMergeWithRandomInitial(seed, data, kernel, True, 10)
    optimalGroupStructure = optimalOkgt.getGroupStructure()
    optimalR2 = optimalOkgt.r2
    estimatedGroupStructures.append(optimalGroupStructure)
    estimatedR2s.append(optimalR2)

pickle.dump(estimatedGroupStructures, open("estimatedGroupStructures.pkl", 'wb'))
pickle.dump(estimatedR2s, open("estimatedR2s.pkl", 'wb'))