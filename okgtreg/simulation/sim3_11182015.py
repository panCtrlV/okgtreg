"""
Simulation for forward selection to determine group structure.
Two models: Wang04 and Wang04WithInteraction

Setting:

    - Number of simulation: 100
    - Sample size 500
"""

import pickle

from okgtreg.DataSimulator import *
from okgtreg.forwardSelection import *

nSim = 100
nSample = 500
seeds = range(100)

kernel = Kernel('gaussian', sigma=0.5)

##################
# Model: Wange04 #
##################
model = "Wang04"
structures = []
r2s = []

for s in seeds:
    # Generate data
    np.random.seed(s)
    y, x = DataSimulator.SimData_Wang04(nSample)
    data = Data(y, x)


    selectionRes = forwardSelection(data, kernel, True, 10)
    structures.append(selectionRes['group'])
    r2s.append(selectionRes['r2'])

pickle.dump(structures, open("./" + model + "-structures.pkl", "wb"))
pickle.dump(r2s, open("./" + model + "-r2s.pkl", "wb"))


###################################
# Model: Wange04 With Interaction #
###################################
model = "Wang04WithInteraction"
structures = []
r2s = []

for s in seeds:
    # Generate data
    np.random.seed(s)
    y, x = DataSimulator.SimData_Wang04WithInteraction(nSample)
    data = Data(y, x)


    selectionRes = forwardSelection(data, kernel, True, 10)
    structures.append(selectionRes['group'])
    r2s.append(selectionRes['r2'])

pickle.dump(structures, open("./" + model + "-structures.pkl", "wb"))
pickle.dump(r2s, open("./" + model + "-r2s.pkl", "wb"))
