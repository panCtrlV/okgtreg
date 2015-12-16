import numpy as np
import itertools
import copy
import warnings
import random


class Group(object):
    def __init__(self, *args, **kwargs):
        """

        :type args: various number of lists of covariates
        :param args: each list is a grouped collection of covariates,
                     e.g. [1], [2,3], [4], [5,6,7], ...

        :type [p]: int, optional
        :param [p]: total number of covariates

        :type [name]: str, optional
        :param [name]: an identity for the current group structure

        :rtype: Group
        :return:
        """

        # group with one covariate must input explicitly

        # Remove any empty groups, e.g. []
        isEmpty = [len(x) != 0 for x in args]
        filteredArgs = tuple(itertools.compress(args, isEmpty))
        n = len(filteredArgs)  # number of non-empty input groups

        # check covariates duplicates
        inputs = [i for g in filteredArgs for i in g]  # flatten args
        uniqueInputs = set(inputs)
        if len(inputs) > len(uniqueInputs):
            raise ValueError("** Each index can only be in one group. "
                             "Please remove duplicates. **")

        # Normalize group structure: check if within and between groups are ordered
        # leadingIndices = [np.array(g).min() for g in filteredArgs]  # list of smallest ind of each group
        leadingIndices = [min(g) for g in filteredArgs]  # this enables string comparison
        isOrdered = all(leadingIndices[i] <= leadingIndices[i+1] for i in xrange(n-1))

        if not isOrdered:
            orders = sorted(range(n), key=lambda k: leadingIndices[k])
            self.partition = tuple(list(np.sort(filteredArgs[order])) for order in orders)
        else:
            self.partition = tuple(list(np.sort(filteredArgs[i])) for i in xrange(n))

        self.size = n

        # accept number of covariates from keyword argument
        # or set automatically as the size of the flattened args if not given
        # if len(kwargs) > 0:
            # for key in ('p'): setattr(self, key, kwargs.get(key))
        if 'p' in kwargs.keys():
            setattr(self, 'p', kwargs.get('p'))
        else:
            self.p = len(inputs)

        if 'name' in kwargs.keys():
            setattr(self, 'name', kwargs.get('name'))
        else:
            self.name = None

        # fields:
        #   partition, size, p, name

    def __getitem__(self, index):
        return self.getPartition(index)

    def __eq__(self, other):
        """
        Compare two Group objects.

        :type other: Group
        :param other: the other group structure to compare to.

        :rtype: boolean
        :return:
        """
        return self.partition == other.partition

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Specify the relative order, strictly smaller, between two Group objects. If each
        group of covariates in `self` is a subset of a group in `other`, and at least one
        is a proper subset, then `self` < `other`. For example, with the following group
        structures:

            g1: ([1], [2,3])
            g2: ([1,4], [2,3])
            g3: ([1], [2,3,4])
            g4: ([1], [2,3], [4])
            g5: ([1], [4], [2,3])

        we have the relationships:

            g1 < g2,
            g1 < g3,
            g1 < g4,
            g1 < g5

        while g4 == g5, so g4 < g5 false False.

        If g1 < g2 and g2 is the true group structure, then g1 is called a "correct"
        in our OKGT paper.

        :type other: Group
        :param other: the other group structure to compare to.

        :rtype: bool
        :return: if `self` is strictly smaller than `other`.
        """
        return np.all([np.any([set(g1) <= set(g2) for g2 in other.partition]) for g1 in self.partition]) and \
               self != other

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __add__(self, other):
        """
        Add two Group objects to return a bigger Group.

        :type other: Group or list
        :param other:

        :type return: Group
        :return:
        """
        if isinstance(other, list):
            otherPartition = (other,)
        elif isinstance(other, tuple):
            otherPartition = other
        else:
            otherPartition = other.partition

        biggerPartition = self.partition + otherPartition
        return Group(*biggerPartition)

    def __str__(self):
        return "%s" % (self.partition,)

    def __repr__(self):
        return "Group structure %s" % (self.partition,)

    def getPartition(self, partitionNumber=None):
        """
        Return one partition from the group structure as a list, e.g. [1] or [1,2].
        The `partitionNumber` start from 1.

        :type partitionNumber: int
        :param partitionNumber: index of the group to extract

        :rtype: list
        :return: group of indices as a list
        """
        if partitionNumber is None:
            return self.partition
        else:
            if partitionNumber <= 0 or partitionNumber > self.size:
                raise ValueError("** \"partitionNumber\" %d is out of bounds. **" % partitionNumber)

            return self.partition[partitionNumber - 1]

    def getPartitions(self, partitionNumbers=None, returnAsGroup=False):
        """
        Return one or more partitions from the current group structure as a tuple,
        e.g. ([1], ), ([1], [2,3])

        :type partitionNumbers: list or None
        :param partitionNumbers:

        :type returnAsGroup: bool
        :param returnAsGroup: if the extracted partitions are extracted as a tuple
                              or a Group object.

        :rtype: tuple or Group
        :return: extracted partitions as a tuple or a Group object.
        """
        if partitionNumbers is None:
            returnPartition = self.partition
        else:
            if np.any([i <=0 or i > self.size for i in partitionNumbers]):
                raise ValueError("** One or more partition numbers are out of bounds. **")
            else:
                returnPartition = tuple([self.partition[i-1] for i in partitionNumbers])

        if returnAsGroup:
            return Group(*returnPartition)
        else:
            return returnPartition

    def getMembership(self, covariateIndex):
        """
        Return which group a given `covariateIndex`-th covariate belongs.
        The first group number is 1. The index for the first covariate is 1.

        :type covaraiteIndex: int
        :param covariateIndex: the index of the covariate whose membership is returned

        :rtype: int
        :return: group membership of the `covariateIndex`-th covariate.
        """
        try:
            return int(np.where([covariateIndex in part for part in self.partition])[0]) + 1
        except TypeError:
            print("** Failed to find covariate %d in the group structure. **" % covariateIndex)

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
        """
        Add a new covariate as a new group in the structure, where
        covariateIndex starts from 1.

        :type covariateIndex: int
        :param covariateIndex: the index of the new covaraite to be added into the exsiting
                               group structure as a new (univariate) group.

        :rtype: Group
        :return: a new group structure with the new covariate added as a new group.
        """
        if covariateIndex in [i for g in self.partition for i in g]:
            raise ValueError("** Covariate %d is already in the partition. **" % covariateIndex)

        return Group(*(self.partition + ([covariateIndex],)) )

    def removeOneCovariate(self, covariateIndex):
        """
        Remove `covariateIndex`-th covariate from the group it belongs
        """
        try:
            ind = int(np.where([covariateIndex in part for part in self.partition])[0])  # where covariateIndex belongs
        except ValueError:
            print("** Covariate %d is not in the group structure. **" % covariateIndex)

        partition = copy.deepcopy(self.partition)
        # We cannot use `partition = self.partition`, since it still reference to `self.partition`.
        # So any change we make on `partition` will also affect `self.partition`.
        partition[ind].remove(covariateIndex)
        # Calling the class name creates a new Group object out of the current scope.
        # Alternatively, we can just call __init__ to change the self object in place.
        # More details can be found at:
        # http://stackoverflow.com/questions/25118798/python-how-to-call-the-constructor-from-within-member-function
        return Group(*partition)

    def removeOneGroup(self, partitionNumber):
        """
        Remove from the current group structure the `groupNumber`-th group and return a new
        Group object.

        :type partitionNumber: int
        :param partitionNumber: the index of the group to be removed. The first group number is 1.

        :rtype: Group
        :return: group structure with one fewer group
        """
        pass

    def has(self, g):
        """
        Check if any group in the group structure has the given
        covariate / set of covariates as a subset. For example, if
        the group structure is given by:

            ([1, 2], [3, 6, 7], [4], [5, 8])

        then the following should be True:

            has([1, 2]), has([6, 7]), has(4), has([4]), has(5)

        :type g: int or list of int
        :param g: the covariate or set of covariates

        :rtype: bool
        :return: if the current group structure has the given covariate(s)
                 as a subset.
        """
        if isinstance(g, int):
            glist = [g]
        else:
            glist = g
        # TODO: Otherwise, I assume g is already a list. Need to check the input.

        if len(glist) > len(set(glist)):
            raise ValueError("** There shouldn't be duplicates in the provided covariate list. **")

        return np.any([set(glist).issubset(set(group)) for group in self.partition])

    def hasAsAGroup(self, g):
        if isinstance(g, int):
            glist = [g]
        else:
            glist = g

        if len(glist) > len(set(glist)):
            raise ValueError("** There shouldn't be duplicates in the provided covariate list. **")

        return g in self.partition

    def _splitOneGroup(self, partitionNumber):
        """
        Completely split a multivariate group in the current group structure
        into univariate groups.

        :type partitionNumber: int
        :param partitionNumber: start from 1

        :rtype: Group
        :return:
        """
        if partitionNumber > self.size or partitionNumber < 1:
            raise ValueError("** \"partitionNumber\" %d is out of bound. **" % partitionNumber)
        else:
            selectedPart = self.getPartition(partitionNumber)
            if len(selectedPart) == 1:
                warnings.warn("** Group %d is univariate. No need to split. **" % partitionNumber)
                # return None
                return self
            else:
                selectedPartAfterSplit = [[i] for i in selectedPart]
                partitionList = list(self.partition)
                partitionList.pop(partitionNumber - 1)
                partitionList.extend(selectedPartAfterSplit)
                return Group(*tuple(partitionList))

    def _splitOneCovariate(self, covariateIndex):
        """
        Split the given covariate into a new univariate group in
        the current group structure.

        :type covariateIndex: int
        :param covariateIndex:

        :rtype: Group
        :return:
        """
        pass

    def _randomSplitOneGroup(self, partitionNumber, seed=None):
        if partitionNumber > self.size or partitionNumber < 1:
            raise ValueError("** \"partitionNumber\" %d is out of bound. **" % partitionNumber)
        else:
            selectedPart = self.getPartition(partitionNumber)  # list
            if len(selectedPart) == 1:
                warnings.warn("** Group %d is univariate. No need to split. **" % partitionNumber)
                return self
            else:
                # Random number generator for picking a covariate
                if seed is None:
                    rg = random.Random()
                else:
                    rg = random.Random(seed)
                chosenCovariate = rg.choice(selectedPart)  # randomly choose a covariate
                return self.removeOneCovariate(chosenCovariate).addNewCovariateAsGroup(chosenCovariate)

    def split(self, partNumber, randomSplit=False, seed=None):
        """
        Split a grouped set of covariates from the current group structure.
        There are three possibilities:
            1) Complete splitting. Each covariate in a group becomes a
               univariate group by itself. For example, [1,2,3] -> [1],[2],[3].
            2) Deterministic splitting. The user specifies the covariate id(s)
               to split and form a new group. For example, given the current group
               [1,2,3], we cah choose to separate 3 from the other two, thus having
               ([1,2],[3]).
            3) Random splitting. A covariate is randomly chosen to be split from a
               group.

        :type partNumber: int
        :param partNumber: which group / part in the current group structure to be split

        :type covariates: int or list of ints
        :param covariates: the covariate or the set of covariates to be separated from
                           the current group, which will form a new group.

        :type randomSplit: bool
        :param randomSplit: whether a covariate is split randomly

        :type seed: int
        :param seed:
        :return:
        """
        # TODO: (2) has not implemented yet.
        if randomSplit:
            return self._randomSplitOneGroup(partNumber, seed)
        else:
            return self._splitOneGroup(partNumber)

    def _mergeTwoGroups(self, partitionNumber1, partitionNumber2):
        """
        Combine `partitionNumber1`-th and `partitionNumber2`-th groups into one group.

        :param partitionNumber1:
        :param partitionNumber2:
        :return:
        """
        # Check if two partition numbers are identical
        if partitionNumber1 == partitionNumber2:
            raise ValueError("** Partition numbers must be different. **")

        # Check if partition numbers are out of bounds
        if np.any(np.array([partitionNumber1, partitionNumber2]) < 1) or \
                np.any(np.array([partitionNumber1, partitionNumber2]) > self.size):
            raise ValueError("** Partition numbers are out of bounds. **")

        # Merger two groups
        part1 = self.getPartition(partitionNumber1)
        part2 = self.getPartition(partitionNumber2)

        remainingParts = list(self.getPartition())
        remainingParts.remove(part1)
        remainingParts.remove(part2)

        mergedPart = part1 + part2

        remainingParts.append(mergedPart)
        return Group(*tuple(remainingParts))


class RandomGroup(Group):
    """
    Given a list of covariate indices or total number of covariates and
    the desired number of groups, randomly partition the covariates into
    groups.
    """
    def __init__(self, size, covariates=None, nCovariates=None, seed=None):
        """
        Two constructors. Either `covariateIndices` or `nCovariates` is provided,
        not both.

        :type size: int
        :param size: desired number of groups in the random group structure

        :type covariates: list
        :param covariates: all covariate indices as a list

        :type nCovariates: int
        :param nCovariates: number of total covariates

        :rtype: Group
        :return: Group object with randomly partitioned structure
        """
        # TODO: Check `covariates` and `nCovariates` cannot be both given.
        # TODO: Partition in to n groups where each has at least one variable.
        if covariates is None:
            covariates = list(np.arange(nCovariates) + 1)

        # Create a random number generator
        if seed is None:
            rg = random.Random()
        else:
            rg = random.Random(seed)

        # Generate random memberships
        # groupIndices = np.random.randint(1, size+1, p)  # size + 1 <= p
                                                        # It is possible that some group number is not sampled
        partition = self._nearEqualRandomPartition(covariates, size, rg)
        Group.__init__(self, *tuple(partition))

    @staticmethod
    def _randomPartition(covariates, n, rg):
        p = len(covariates)
        groupIndices = np.array([rg.choice(range(n)) for i in range(p)]) + 1
        indexedCovariates = zip(groupIndices, covariates)
        indexedCovariates.sort(key=lambda x: x[0])
        partition = []
        for k, g in itertools.groupby(indexedCovariates, lambda x: x[0]):
            partition.append([pair[1] for pair in g])
        return partition

    @staticmethod
    def _nearEqualRandomPartition(covariates, n, rg):
        """
        Randomly partition covariates into groups of nearly equal size.

        :type n: int
        :param n: number of groups

        :type rg: random.Random
        :param rg:

        :rtype: list
        :return:
        """
        covariateList = covariates
        rg.shuffle(covariateList)
        division = len(covariateList) / float(n)
        return [ covariateList[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]


if __name__=='__main__':
    randomGroup = RandomGroup(size=4, covariateIndices=[1,2,3,4,5,6,7,8,9,10])
    randomGroup

    randomGroup = RandomGroup(size=4, nCovariates=10)
    randomGroup

