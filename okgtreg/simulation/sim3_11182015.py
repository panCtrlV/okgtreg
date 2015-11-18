import pickle

from okgtreg.DataSimulator import *
# from okgtreg.Data import *
# from okgtreg.Kernel import *
from okgtreg.forwardSelection import *


"""
Simulation for forward selection to determine group structure.
Two models: Wang04 and Wang04WithInteraction

Setting:

    - Number of simulation: 100
    - Sample size 500
"""


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

for s in seeds:  # had problem at seed=21
    # Generate data
    np.random.seed(s)
    y, x = DataSimulator.SimData_Wang04WithInteraction(nSample)
    data = Data(y, x)

    selectionRes = forwardSelection(data, kernel, True, 10)
    structures.append(selectionRes['group'])
    r2s.append(selectionRes['r2'])

pickle.dump(structures, open("./" + model + "-structures.pkl", "wb"))
pickle.dump(r2s, open("./" + model + "-r2s.pkl", "wb"))


"""
The following example will run into error

    nSim = 100
    nSample = 500
    kernel = Kernel('gaussian', sigma=0.5)

    s = 21
    np.random.seed(s)
    y, x = DataSimulator.SimData_Wang04WithInteraction(nSample)
    data = Data(y, x)
    selectionRes = forwardSelection(data, kernel, True, 10)

It is because while reaching the stage when [6, 7] are the only
two covarates to be added to the structure, none of them would
improve R2 for OKGT.
"""