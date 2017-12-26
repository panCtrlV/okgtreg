__author__ = 'panc'

'''
Process the simulation results from "validate_backward.py" for Model 3.
'''

from okgtreg.simulation.sim_02052016.validate_analyze_utility import *
from okgtreg.groupStructureDetection.backwardPartition import rkhsCapacity

# Model 2 group structure
trueGroupStructure = Group([1], [2], [3], [4], [5], [6])

cwd = "okgtreg/simulation/sim_02052016/validate_backward_model3"

# Try one simulation
## Create an object
fname = "validate-model3-seed1-201602090712.pkl"
with open(cwd + '/' + fname, 'rb') as f:
    sim1 = ValidateResultProcessor(f, trueGroupStructure)

## The selected group structure from 1-fold validation
print sim1.bestGroupStructure()

## What are the selected group structures from training phase
## for different (mu, alpha)
for g in sim1.selectedGroupStructuresFromTrain():
    print g

## Rank of those group structures after test based on the
## test R2, and penalized test R2
testRes = sim1.res['test']
testResWithoutPenalty = collections.defaultdict(dict)
testResWithPenalty = collections.defaultdict(dict)
for k, v in testRes.iteritems():
    testResWithoutPenalty[v['r2']] = v['group']
    penalty = k[0] * rkhsCapacity(Group(group_struct_string=v['group']), k[1])  # v['group'] is a string
    pR2 = v['r2'] - penalty
    testResWithPenalty[v['r2'] - penalty] = v['group']

testResWithoutPenalty

testResWithPenalty

sorted(testResWithPenalty, key=testResWithPenalty.get, reverse=True)

## For the true group structures, what are the (mu, alpha) values
optimalParameterValues = sim1.parameterValuesForTrueGroupStructure()

# For all 100 simulations for Model 3
## Create object
allRes_model3 = ValidateResultProcessorAll(cwd, trueGroupStructure)

## print bestGroupStructures and their frequency
bestGroupStructureFreq = allRes_model3.bestGroupStructureFrequency()
## The true group structure is not selected using 1-fold validation

## For each (mu, alpha), what is the frequency of selecting the true
## group structure from train set. Has nothing to do with test R2.
allRes_model3.consistentSelectionFrequencyForParameters()

# How about adding the corresponding penalty to the test R2 ?
