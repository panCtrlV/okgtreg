from okgtreg.OKGTReg import *
from okgtreg.Parameters import *


"""
Determining group structure by forward selection.
"""

# kernel = Kernel('gaussian', sigma=0.5)

def forwardInclusion(data, kernel, method='vanilla', rank=10, seed=None, lmbda=1e-5):
    """
    Forward selection procedure which detects group structure for OKGT.
    It is assumed that:
     1) all covariates are used
     2) same kernel function (including parameters) for all groups

    By using the forward selection procedure, the algorithm starts with
    an empty group structure. During each iteration, one covariate is added,
    either as a univariate group or in an existing group, whichever resulting
    in larger increase in R2.

    For more details of the algorithm, please refer to my notes.

    :type data: Data
    :param data: data whose group structure to be determined

    :type kernel: Kernel
    :param kernel:  kernel function, sane for all groups in the group
                    structure

    :type useLowRankApproximation: bool
    :param useLowRankApproximation: flag for low rank approximation of kernel matrices.
                                    If True, each kernel matrices are approximated by its
                                    low rank counterpart. Currently, only Nystroem method
                                    is implemented.

    :type rank: int
    :param rank: number of ranks for the low rank approximation.

    :type seed: int
    :param seed: seed for Nystroem method of low-rank matrix approximation for
                 kernel matrices.

    :rtype: Group
    :return: selected group structure
    """
    covariatesPool = list(np.arange(data.p) + 1)
    oldGroup = Group()  # start with an empty group structure
    # bestR2 = 0.
    bestR2 = -np.inf
    bestCovariateIndex = None

    while len(covariatesPool):
        print "** Available covariates: ", covariatesPool
        # add a new group no matter what
        print "** Add as a new group: **"
        for covariateInd in covariatesPool:
            print("\t try covariate %d ..." % covariateInd)
            currentGroup = oldGroup.addNewCovariateAsGroup(covariateInd)
            # print("\t\t current group structure: %s " % (currentGroup.getPartitions(),))
            # The following OKGT needs a subset of data and the grouped covariate
            # indices being **normalized**, so that the training is done as if we are
            # using a complete data.
            dataForOKGT, groupForOKGT = data.getGroupedData(currentGroup)
            # print groupForOKGT.partition
            # TODO: Currently, using same kernel for all groups.
            # todo: Is it possible to adapt kernels to different group structure?
            parametersForOKGT = Parameters(groupForOKGT, kernel, [kernel]*groupForOKGT.size)
            currentOKGT = OKGTReg(dataForOKGT, parametersForOKGT)
            # Train OKGT
            res = currentOKGT.train(method, rank, seed)
            # currentR2 = res['r2']
            capacity = sum([len(g) ** len(g) for g in currentGroup.partition])
            print("\t\t current group structure: %s with capacity: %d" %
                  (currentGroup.getPartitions(), capacity))
            currentR2 = res['r2'] - lmbda * capacity
            # Check if there is improvement
            if currentR2 > bestR2:
                # print("\t\t current R2 =\t %.10f \t *" % currentR2)
                print("\t\t current R2 (penalized) =\t %.10f \t *" % currentR2)
                bestR2 = currentR2
                bestCovariateIndex = covariateInd
                newGroup = currentGroup
            else:
                # print("\t\t current R2 =\t %.10f" % currentR2)
                print("\t\t current R2 (penalized) =\t %.10f" % currentR2)

            # print("\t\t best R2 =\t\t %.10f" % bestR2)
            print("\t\t best R2 (penalized) =\t\t %.10f" % bestR2)

        print("** updated group structure is: %s \n" % (newGroup.getPartitions(), ))
        # If group structure is not empty, a new covariate can be added to an existing group
        if oldGroup.size > 0:
            print "** Add to an existing group: **"
            # can add new covariate to existing group
            for covariateInd in covariatesPool:  # pick a covariate
                print("\t try adding covariate %d " % covariateInd)
                for groupInd in np.arange(oldGroup.size)+1:  # pick an existing group
                    # print oldGroup.partition
                    print("\t in group %d ..." % groupInd)
                    currentGroup = oldGroup.addNewCovariateToGroup(covariateInd, groupInd)
                    # print("\t\t current group structure: %s " % (currentGroup.getPartitions(),))
                    # print currentGroup.partition
                    dataForOKGT, groupForOKGT = data.getGroupedData(currentGroup)
                    # print groupForOKGT.partition
                    # xkernels = [kernel] * groupForOKGT.size
                    parametersForOKGT = Parameters(groupForOKGT, kernel, [kernel]*groupForOKGT.size)
                    currentOKGT = OKGTReg(dataForOKGT, parametersForOKGT)
                    # Train OKGT
                    res = currentOKGT.train(method=method, nComponents=rank, seed=seed)
                    # currentR2 = res['r2']
                    capacity = sum([len(g) ** len(g) for g in currentGroup.partition])
                    print("\t\t current group structure: %s with capacity: %d" %
                          (currentGroup.getPartitions(), capacity))
                    currentR2 = res['r2'] - lmbda * capacity
                    # Check if there is improvement
                    if currentR2 > bestR2:
                        # print("\t\t current R2 =\t %.10f \t *" % currentR2)
                        print("\t\t current R2 (penalized) =\t %.10f \t *" % currentR2)
                        bestR2 = currentR2
                        bestCovariateIndex = covariateInd
                        # bestGroupIndex = groupInd
                        newGroup = currentGroup
                    else:
                        # print("\t\t current R2 =\t %.10f" % currentR2)
                        print("\t\t current R2 (penalized) =\t %.10f" % currentR2)

                    # print("\t\t best R2 =\t\t %.10f" % bestR2)
                    print("\t\t best R2 (penalized) =\t\t %.10f" % bestR2)
        # Add early termination if no further improvement
        # TODO: It is possible that there is no further improvement by adding new
        # todo: covariates, but there are still covariates in the pool. Currently,
        # todo: I use early termination. That is, if an iteration does not add a
        # todo: new covariate into the existing group structure, the algorithm stops.
        if newGroup == oldGroup:
            print("** Early termination: %s are still avaliable, "
                  "but no further improvement. ** \n" % covariatesPool)
            break
        else:
            print("** updated group structure is: %s \n" % (newGroup.getPartitions(), ))
            covariatesPool.remove(bestCovariateIndex)  # TODO: update in-place, good?
            oldGroup = newGroup

    print ("** SELECTED GROUP STRUCTURE: %s ** \n" % (oldGroup.partition, ))
    # return oldGroup
    # return dict(group=oldGroup, r2=bestR2)
    return dict(group=oldGroup, r2p=bestR2)


if __name__ == '__main__':
    # Model
    def model10(n):
        x = np.random.uniform(1, 2, (n, 6))
        e = np.random.normal(0., 0.1, n)
        y = np.log(1. +
                   np.log(x[:, 0]) +
                   x[:, 1] / np.exp(x[:, 2]) +
                   np.power(x[:, 3] + x[:, 4], x[:, 5]) + e)
        return Data(y, x), Group([1], [2, 3], [4, 5, 6])


    # Simulate data
    n = 500
    np.random.seed(0)
    data, tgroup = model10(500)

    print tgroup

    # Run forward inclusion method for group structure detection
    kernel = Kernel('gaussian', sigma=0.5)
    res = forwardInclusion(data, kernel)
