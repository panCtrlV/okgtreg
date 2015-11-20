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


"""Simmulation result

"""
import pickle

structuresWang04 = pickle.load(open("/Users/panc25/sshfs_map/Wang04-structures.pkl"))
structuresWang04

trueGroup = Group([1], [2], [3], [4], [5])
np.array([group == trueGroup for group in structuresWang04]).sum() / 100.


structuresWang04WithInteraction = pickle.load(open("/Users/panc25/sshfs_map/Wang04WithInteraction-structures.pkl"))
structuresWang04WithInteraction
trueGroup = Group([1], [2], [3], [4], [5], [6, 7])
np.array([group == trueGroup for group in structuresWang04WithInteraction]).sum() / 100.

def hasGroup(groupList, groupStructure):
    return groupList in groupStructure.partition

np.sum([hasGroup([6,7], group) for group in structuresWang04WithInteraction]) / 100.


def hasCovaraitesAsGroup(): pass  # [6,7] should be a subset of a group in the selected group structure



s = 21
np.random.seed(s)
nSample = 500
y, x = DataSimulator.SimData_Wang04WithInteraction(nSample)
data = Data(y, x)

group = Group([1], [2, 5], [3], [4], [6, 7])
kernel = Kernel('gaussian', sigma=0.5)
ykernel = kernel
xkernels = [kernel]*group.size
parameters = Parameters(group, ykernel, xkernels)


# No low rank
okgt = OKGTReg(data, parameters)
res = okgt.train_Vanilla()
res['r2']

# With low rank
res2 = okgt.train_Nystroem(10)
res2['r2']