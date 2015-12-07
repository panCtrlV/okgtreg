"""
This simulation test if the Split and Merger with Random Initial can recover
the true group structure.

The model used here is:

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
import os

from okgtreg.DataSimulator import DataSimulator
from okgtreg.Kernel import Kernel
from okgtreg.groupStructureDetection.splitAndMergeWithRandomInitial import splitAndMergeWithRandomInitial


nSim = 100
seeds = range(nSim)

# Same data
n = 500

np.random.seed(25)
data = DataSimulator.SimData_Wang04WithInteraction2(500)

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

# Save results
dir = os.path.dirname(os.path.realpath(__file__))
pickle.dump(estimatedGroupStructures, open(dir + "/estimatedGroupStructures.pkl", 'wb'))
pickle.dump(estimatedR2s, open(dir + "/estimatedR2s.pkl", 'wb'))


# ----------------
# Take one sample
# ----------------
seed = 0



