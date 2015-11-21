import numpy as np
import itertools
import copy


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

    def getPartition(self, partitionNumber=None):
        """
        Return one partition from the group structure as a list, e.g. [1] or [1,2].
        The `partitionNumber` start from 1.
        """
        if partitionNumber is None:
            return self.partition
        else:
            if partitionNumber <= 0 or partitionNumber > self.size:
                raise ValueError("** 'partitionNumber' is out of bounds. **")

            return self.partition[partitionNumber - 1]

    def getPartitions(self, partitionNumbers=None, returnAsGroup=False):
        """
        Return one or more partitions from the current group structure as a tuple,
        e.g. ([1], ), ([1], [2,3])

        :type partitionNumbers: list or None
        :param partitionNumbers:
        :return:
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
        The first group number is 1.
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
        Add a new covariate as a new group in the structure
        covariateIndex starts from 1
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

    def removeOneGroup(self, groupNumber):
        """
        Remove from the current group structure the `groupNumber`-th group and return a new
        Group object.

        :type groupNumber: int
        :param groupNumber: the index of the group to be removed. The first group number is 1.

        :type return: Group
        :return: group structure with one fewer group
        """
        pass

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

    def __str__(self):
        return("%s" % (self.partition,))

    def __repr__(self):
        return("Group structure %s" % (self.partition,))


# Test removeOneCovariate
# g = Group([1], [2,3])
# print g
# g2 = g.removeOneCovariate(1)  # `removeOneCovariate` creates a new Group object
# print g2
# print g

# Test getPartitions
# g = Group([1,4], [2,3], [5], [8,7,10], [6, 9])
# g
# g.getPartitions([1,2])  # get two groups
# g.getPartitions([1])   # get only one group
# g.getPartitions([2, 4])  # two groups apart from each other
# g.getPartitions([g.size + 1])  # out of bounds, fail

# g.getPartitions([1,2], True)
# g.getPartitions([1], True)   # get only one group
# g.getPartitions([2, 4], True)  # two groups apart from each other
# g.getPartitions([g.size + 1], True)  # out of bounds, fail

# Test `__add__` overriding
# Group + Group
# g1 = Group([1], [3,4])
# g2 = Group([2], [5,6])
# g1 + g2
#
# g1 = Group([1], [3,4])
# g2 = Group([2], [3,5])
# g1 + g2  # fail because 3 occurs twice

# Group + list
# g1 = Group([1], [3,4])
# g1 + [2,5]

# Group + tuple
# g1 = Group([1], [3,4])
# g1 + ([2,5], [6,7])


# Test `has`
# g = Group([1, 2], [3, 6, 7], [4], [5, 8])
# g.has(1)
# g.has([1])
# g.has([6, 7])
# g.has([6, 8])
# g.has([3])

# Test `hasAsAGroup`
# g = Group([1, 2], [3, 6, 7], [4], [5, 8])
# g.hasAsAGroup([1])

g1 = Group([1], [2,3])
g2 = Group([1,4], [2,3])
# g3 = Group([1], [2,3,4])
# g4 = Group([1], [2,3], [4])
# g5 = Group([1], [4], [2,3])
#
# print g1 < g2
# print g1 < g3
# print g1 < g4
# print g1 < g5
# print g2 < g3
# print g3 < g2
# print g4 <= g5, ": ", g4, "<=", g5
# print g4 < g5, ": ", g4, "<", g5


