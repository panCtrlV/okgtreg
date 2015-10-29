class Group(object):
    def __init__(self, *args, **kwargs):
        # group with one covariate must input explicitly

        n = len(args)  # number of input groups

        # check duplicates
        inputs = [i for g in args for i in g]
        uniqueInputs = set(inputs)
        if len(inputs) > len(uniqueInputs):
            raise ValueError("** Each index can only be in one group. "
                             "Please remove duplication. **")

        # check if groups are ordered by the first index
        firstIndices = [g[0] for g in args]
        isOrdered = all(firstIndices[i] <= firstIndices[i+1] for i in xrange(n - 1))

        if not isOrdered:
            orders = sorted(range(n), key=lambda k: firstIndices[k])
            self.partition = tuple(args[order] for order in orders)
        else:
            self.partition = args

        self.size = n

        # accept number of covariates or set automatically if not given
        if len(kwargs) > 0:
            for key in ('p'): setattr(self, key, kwargs.get(key))
        else:
            self.p = len(inputs)

    def getPartition(self, partitionNumber=None):
        if partitionNumber is None:
            return self.partition
        else:
            if partitionNumber <= 0 or partitionNumber > self.size:
                raise ValueError("** 'partitionNumber' is is out of bounds. **")

            return self.partition[partitionNumber - 1]