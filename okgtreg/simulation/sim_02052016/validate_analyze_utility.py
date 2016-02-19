__author__ = 'panc'

import pickle
import os
import collections
import glob

from okgtreg.Group import Group


# The following class processes one .pkl file
class ValidateResultProcessor(object):
    def __init__(self, f, true_group_structure=None):
        """

        :type f: file object
        :param f:
        :param true_group_structure:
        :return:
        """
        self.res = pickle.load(f)
        self.true_group_structure = true_group_structure

    # Find the best group structure from one simulation
    def bestGroupStructure(self):
        lookupDict = self.res['lookup']
        return max(lookupDict, key=lookupDict.get)

    # List all selected group structures from a train set
    # under different (mu, alpha)
    def selectedGroupStructuresFromTrain(self):
        trainRes = self.res['train']
        gstructs = []
        for k, v in trainRes.iteritems():
            gstructs.append(v['group'].__str__())
        return set(gstructs)

    # For each selected group structure, what are the corresponding
    # (mu, alpha) values
    def selectedGroupStructuresWithTuningParameters(self):
        selectedGroupStructuresWithTuningParameters = collections.defaultdict(list)
        for k, v in self.res['train'].iteritems():
            selectedGroupStructuresWithTuningParameters[v['group'].__str__()].append(k)
        return selectedGroupStructuresWithTuningParameters

    # What are the (mu, alpha) values that selected the true
    # group structures from the train set. Has nothing to do
    # with the test R2.
    def parameterValuesForGroupStructure(self, g):
        parameterValuesForGroupStructure = self.selectedGroupStructuresWithTuningParameters()[g.__str__()]
        print 'mu : alpha'
        for item in parameterValuesForGroupStructure:
            print item[0], ' , ', item[1]
        return parameterValuesForGroupStructure

    def parameterValuesForTrueGroupStructure(self):
        return self.parameterValuesForGroupStructure(self.true_group_structure)


# The following class process all simulations
# for one model. The input is the directory where
# (only) .pkl files are stored.
class ValidateResultProcessorAll(object):
    def __init__(self, directory, true_group_structure=None):
        allResults = []
        # for fname in os.listdir(directory):
        for filepath in glob.glob(directory + '/' + '*.pkl'):
            print filepath
            with open(filepath, 'rb') as f:
                allResults.append(ValidateResultProcessor(f))
        self.allResults = allResults  # list of ValidateResultProcessor's for all simulations
        self.true_group_structure = true_group_structure

    # List the best group structures in terms of test R2
    # and their frequency in all the simulations
    def bestGroupStructureFrequency(self):
        bestGroupStructures = [res.bestGroupStructure() for res in self.allResults]
        counter = collections.Counter(bestGroupStructures)
        for k, v in counter.most_common():
            print k, ' : ', v
        return counter

    # What is the % of time each (mu, alpha) selects the true
    # group structure
    def consistentSelectionFrequencyForParameters(self):
        optimalParameterValues = []
        for res in self.allResults:
            optimalParameterValues += res.parameterValuesForGroupStructure(self.true_group_structure)
        return collections.Counter(optimalParameterValues)
