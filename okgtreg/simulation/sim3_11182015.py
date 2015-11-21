from okgtreg.DataSimulator import *
from okgtreg.forwardSelection import *
# from okgtreg.Data import *
# from okgtreg.Kernel import *

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

# Process simulation results
import pickle
import matplotlib.pyplot as plt


simulationDir = "/Users/panc25/Dropbox/Research/Zhu_Michael/my_paper/paper_OKGT/simulation/okgt_python_11182015"


# Read in the forward selection sim results for Wang04 (fully additive)
structuresWang04 = pickle.load(open(simulationDir + "/Wang04-structures.pkl"))
trueGroup = Group([1], [2], [3], [4], [5])
# Success rate if we only consider perfect match
perfectRate = np.array([group == trueGroup for group in structuresWang04]).sum() / 100.
print("Rate of perfect match is %.04f%%" % perfectRate)

# Read in the forward selection sim results for Wange04WithInteraction (one interaction)
structuresWang04WithInteraction = pickle.load(open(simulationDir + "/Wang04WithInteraction-structures.pkl"))
# If only consider "correctness"
successRate = np.sum([group.has([6, 7]) for group in structuresWang04WithInteraction]) / 100.
print("Percent of correct recovery is %.04f%%" % successRate)
# If consider perfect match
trueGroup = Group([1], [2], [3], [4], [5], [6, 7])
perfectRate = np.sum(group == trueGroup for group in structuresWang04WithInteraction) / 100.
print("Percent of perfect match is %.04f%%" % perfectRate)


# Read in R2 for Wang04
r2Wang04 = pickle.load(open(simulationDir + "/Wang04-r2s.pkl"))
print("Averge R2 = %.04f" % np.mean(r2Wang04))
print("Range of R2 = [%.04f, %.04f]" % (np.min(r2Wang04), np.max(r2Wang04)))
print("Standard deviation of R2 = %.04f" % np.std(r2Wang04))

plt.boxplot(np.array(r2Wang04))

# Read in R2 for Wang04WithInteraction
r2Wang04WithInteraction = pickle.load(open(simulationDir + "/Wang04WithInteraction-r2s.pkl"))
print("Averge R2 = %.04f" % np.mean(r2Wang04WithInteraction))
print("Range of R2 = [%.04f, %.04f]" % (np.min(r2Wang04WithInteraction), np.max(r2Wang04WithInteraction)))
print("Standard deviation of R2 = %.04f" % np.std(r2Wang04WithInteraction))

plt.boxplot(np.array(r2Wang04WithInteraction))


"""Simulation result
1. When the model is Wang04, i.e. fully additive without noise, 87% of the time the true structure was recovered.
   However, this rate may not be useful for OKGT because OKGT aims at finding a "correct" group structure. That is,
   as long as a set of covariates from the same group in the true group structure is a subset of a group in the
   estimated group structure, the estimate is considered successful. If we evaluate the estimation performance
   according to this criterion, then any estimate from a Wang04 data is "correct".

   In terms of R2, the average is 0.9901, with the range [0.9464, 0.9924] and std dev 0.0049.

2. The simulation results for Wang04WithInteraction is more promising. If we consider a "correct" group structure
   as a successful estimate, the success percentage is 97%.

   If we consider a perfect match being successful, then the percentage is 50%.

   In terms of R2, the average is 0.9806, with the range [0.9027, 0.9927] and std dev 0.0176.
"""


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