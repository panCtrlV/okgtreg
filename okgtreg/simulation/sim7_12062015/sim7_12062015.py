"""
This simulation tests the average performance of Split and Merge with Random Initial
on group structure recovery.

It is different from sim5 and sim6 in that the data sets are different for each simulation,
while the seed for Nystroem low rank approximation remains the same for all simulations.

The model used here is that same as that in sim5, i.e.

        y = np.log(4.0 +
                   np.sin(4 * x[:, 0]) +
                   np.abs(x[:, 1]) +
                   x[:, 2]**2 +
                   x[:, 3]**3 +
                   x[:, 4] +
                   abs(x[:, 5] * x[:, 6] * x[:, 7]) +
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

# Same kernel
kernel = Kernel('gaussian', sigma=0.5)

# Empty lists to save simulation results
estimatedGroupStructures = []
estimatedR2s = []

for seed in seeds:
    np.random.seed(seed)
    data = DataSimulator.SimData_Wang04WithInteraction2(500)

    optimalOkgt = splitAndMergeWithRandomInitial(25, data, kernel, True, 10)  # seed fixed for Nystroem
    optimalGroupStructure = optimalOkgt.getGroupStructure()
    optimalR2 = optimalOkgt.r2
    estimatedGroupStructures.append(optimalGroupStructure)
    estimatedR2s.append(optimalR2)

pickle.dump(estimatedGroupStructures, open("estimatedGroupStructures.pkl", 'wb'))
pickle.dump(estimatedR2s, open("estimatedR2s.pkl", 'wb'))