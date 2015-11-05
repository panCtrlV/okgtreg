import numpy as np
import itertools


class Group(object):
    def __init__(self, *args, **kwargs):
        # group with one covariate must input explicitly

        # Remove any empty groups
        isEmpty = [len(x) != 0 for x in args]
        filteredArgs = tuple(itertools.compress(args, isEmpty))

        n = len(filteredArgs)  # number of non-empty input groups

        # check duplicates
        inputs = [i for g in filteredArgs for i in g]  # flatten args
        uniqueInputs = set(inputs)
        if len(inputs) > len(uniqueInputs):
            raise ValueError("** Each index can only be in one group. "
                             "Please remove duplicates. **")

        # Normalize group structure:
        # check if within and between groups are ordered
        leadingIndices = [np.array(g).min() for g in filteredArgs]  # list of smallest ind of each group
        isOrdered = all(leadingIndices[i] <= leadingIndices[i+1] for i in xrange(n-1))

        if not isOrdered:
            orders = sorted(range(n), key=lambda k: leadingIndices[k])
            self.partition = tuple(list(np.sort(filteredArgs[order])) for order in orders)
        else:
            self.partition = tuple(list(np.sort(filteredArgs[i])) for i in xrange(n))

        self.size = n

        # accept number of covariates from keyword argument
        # or set automatically as the size of the flattened args if not given
        if len(kwargs) > 0:
            for key in ('p'): setattr(self, key, kwargs.get(key))
        else:
            self.p = len(inputs)

        # fields:
        #   partition, size, p

    def getPartition(self, partitionNumber=None):
        # partitionNumber start from 1
        if partitionNumber is None:
            return self.partition
        else:
            if partitionNumber <= 0 or partitionNumber > self.size:
                raise ValueError("** 'partitionNumber' is is out of bounds. **")

            return self.partition[partitionNumber - 1]

    def __getitem__(self, index):
        return self.getPartition(index)

    def addNewCovariateToGroup(self, covariateIndex, groupNumber):
        # Add a new covariate to an existing group in the structure
        # Both arguments start from 1
        if groupNumber > self.size:
            raise ValueError("** 'groupNumber' = %d is out of bound. "
                             "Partition has %d group(s). **" % (groupNumber, self.size))

        if covariateIndex in [i for g in self.partition for i in g]:
            raise ValueError("** Covariate %d is already in the partition. **" % covariateIndex)

        updatedPart = self.partition[groupNumber-1] + [covariateIndex]
        partitionList = list(self.partition)
        partitionList[groupNumber - 1] = updatedPart
        updatedPartition = tuple(partitionList)
        return Group(*updatedPartition)

        # print "updatedGroup: ", updatedGroup
        #
        # print "self.partition = ", self.partition
        # print "self.size = ", self.size
        # print "groupNumber = ", groupNumber
        # print [self.partition[i] for i in range(self.size) if i is not groupNumber-1]
        # unchangedGroups = tuple(self.partition[i] for i in range(self.size) if i is not groupNumber-1)
        # print "unchangedGroups: ", unchangedGroups
        # return Group(*(unchangedGroups + (updatedGroup,)))

    def addNewCovariateAsGroup(self, covariateIndex):
        # Add a new covariate as a new group in the structure
        # covariateIndex starts from 1
        if covariateIndex in [i for g in self.partition for i in g]:
            raise ValueError("** Covariate %d is already in the partition. **" % covariateIndex)

        return Group(*(self.partition + ([covariateIndex],)) )

    def removeOneCovariate(self, covariateIndex):
        # Remove `covariateIndex`-th covariate from the group it belongs
        try:
            ind = int(np.where([covariateIndex in part for part in self.partition])[0])  # where covariateIndex belongs
        except ValueError:
            print("** Covariate %d is not in the group structure. **" % covariateIndex)

        self.partition[ind].remove(covariateIndex)
        return Group(*self.partition)