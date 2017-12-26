import numpy as np
import copy

from okgtreg.Group import Group
from okgtreg.OKGTReg import OKGTReg2
from okgtreg.Parameters import Parameters


"""
Determining group structure by backward stepwise selection.
In doing so, we first start with a fully non-parametric model, 
i.e.

    g(y) = f(x_1, x_2, ..., x_p)

and fit the model and record its R2 (goodness of fit measure).

The group (x_1, x_2, ..., x_p) serves as the pool of available variables 
(which can be moved around in the current group structure. At the 
beginning of the algorithm, all predictor variables are included in the pool). 
For each variable in the pool, we apply the following operations in order:

1. Create a new group using the variable on its own. Then, the resulting 
    (new) group structure is used to fit by OKGT.

2. If there are one or more groups other than the "pool" we
   started from, the selected variable joins each
   of the other groups to form a new/different group structure,
   which is used to fit an OKGT.

During each operation, the R2 from the new group structure is compared 
with that from the existing group structure, to see if there is an improvement
(i.e. R2 increases). If R2 increases, the new group structure is accepted, and
the corresponding R2 is recorded.

The above procedure continues until one of the following conditions is met:

1. R2 stops increasing (even if there are still multiple variables in the pool)
2. There is only one variable remaining in the pool 

**Note** 
The second condition above helps to avoid redundancy. For example,
if there are two variables in total (x1, x2). The only two possible
group structures are [(x1, x2)] and [(x1), (x2)]. That is,
once a variable is separated from the other, the algorithm
is done. There is no need to test for the last variable in the pool.
"""


from okgtreg.gasi.utility import rkhsCapacity

# def rkhsCapacity(group, alpha):
#     return sum([alpha ** len(g) for g in group.partition])

# ------------------------
# Start forward selection
# ------------------------
def backwardPartition(data, kernel, method='vanilla', rank=10, seed=None,
                      mu=1e-4, alpha=np.e, logger=None):
    '''
    Backward stepwise algorithm for identifying group additive structure.

    :param data:
    :param kernel:
    :param method:
    :param rank:
    :param seed:
    :param mu: the first tuning parameter for the capacity penalty $\lambda * a^d$
    :param alpha: the second tuning parameter for capacity penalty in $\lambda * a^d$
    :param logger:
    :return:
    '''
    covariates_pool = list( np.arange(data.p) + 1 )
    oldGroup = Group(covariates_pool)  # start with a large single group
    p = oldGroup.p
    # bestR2 = 0.
    bestR2 = -np.inf
    bestCovariateIndex = None

    ctr = 0
    while len(covariates_pool) > 1:
        ctr += 1
        print("** === Step %d === **" % ctr)
        # Create a new group
        print("** Create a new group: **")
        for covariateInd in covariates_pool:
            print("\t Create a new group for covariate %d ..." % covariateInd)
            _currentGroup = oldGroup.removeOneCovariate(covariateInd)
            currentGroup = _currentGroup.addNewCovariateAsGroup(covariateInd)
            # print("\t\t current group structure: %s " % (currentGroup.getPartitions(),))
            # Contrary to forward selection, the data matrix doesn't
            # change.
            parameters = Parameters(currentGroup, kernel, [kernel]*currentGroup.size)
            currentOKGT = OKGTReg2(data, parameters)
            # Train OKGT
            res = currentOKGT.train(method, rank, seed)
            # currentR2 = res['r2']
            # capacity = sum([len(g) ** len(g) for g in currentGroup.partition])
            capacity = rkhsCapacity(currentGroup, alpha)
            print("\t\t current group structure: %s with capacity: %.04f" %
                  (currentGroup.getPartitions(), capacity))
            currentR2 = res['r2'] - mu * capacity
            if currentR2 > bestR2:
                # print("\t\t current R2 =\t %.10f \t *" % currentR2)
                print("\t\t current R2 (penalized) =\t %.10f \t *" % currentR2)
                bestR2 = currentR2
                newGroup = currentGroup
                bestCovariateIndex = covariateInd
            else:
                # print("\t\t current R2 =\t %.10f" % currentR2)
                print("\t\t current R2 (penalized) =\t %.10f" % currentR2)

            # print("\t\t best R2 =\t\t %.10f \n" % bestR2)
            print("\t\t best R2 (penalized) =\t\t %.10f \n" % bestR2)

        # print("** Updated group structure is: %s \n" % (newGroup.partition, ))
        # print '\n'
        # If there are already new groups, a chosen variable can join one of the
        # new groups instead of creating a new group.
        print "** Add to an existing group: **"
        if oldGroup.size > 1:
            for covariateInd in covariates_pool:
                print("\t try adding covariate %d " % covariateInd)
                # Remove `covariateInd`-th covariate from the pool,
                # which will be added into one of the other groups.
                updatedCovariatesPool = copy.deepcopy(covariates_pool)
                updatedCovariatesPool.remove(covariateInd)
                # Get the group number of the chosen `covariateInd`
                covariateMember = oldGroup.getMembership(covariateInd)
                # Take all other groups as a Group object
                otherGroupInds = list(np.arange(oldGroup.size)+1)
                otherGroupInds.remove(covariateMember)

                # print type(otherGroupInds), ": ", otherGroupInds

                otherGroup = oldGroup.getPartitions(otherGroupInds, True)
                # Try adding the chosen `covariateInd` to each of the other groups
                for groupInd in np.arange(otherGroup.size) + 1:
                    print("\t   in other group %d ..." % groupInd)
                    updatedOtherGroup = otherGroup.addNewCovariateToGroup(covariateInd, groupInd)
                    currentGroup = updatedOtherGroup + updatedCovariatesPool
                    print("\t\t current group structure: %s " % (currentGroup.getPartitions(),))
                    parameters = Parameters(currentGroup, kernel, [kernel]*currentGroup.size)
                    currentOKGT = OKGTReg2(data, parameters)
                    # Train OKGT
                    res = currentOKGT.train(method, rank, seed)
                    # currentR2 = res['r2']
                    # capacity = sum([len(g) ** len(g) for g in currentGroup.partition])
                    capacity = rkhsCapacity(currentGroup, alpha)
                    print("\t\t current group structure: %s with capacity: %.04f" %
                          (currentGroup.getPartitions(), capacity))
                    currentR2 = res['r2'] - mu * capacity
                    # Check if there is improvement
                    if currentR2 > bestR2:
                        # print("\t\t current R2 =\t %.10f \t *" % currentR2)
                        print("\t\t current R2 (penalized) =\t %.10f \t *" % currentR2)
                        bestR2 = currentR2
                        newGroup = currentGroup
                        bestCovariateIndex = covariateInd
                    else:
                        # print("\t\t current R2 =\t %.10f" % currentR2)
                        print("\t\t current R2 (penalized) =\t %.10f" % currentR2)

                    # print("\t\t best R2 =\t\t %.10f \n" % bestR2)
                    print("\t\t best R2 (penalized) =\t\t %.10f \n" % bestR2)
        else:
            print("\t ** No other groups than the pool. Pass ... ** \n")

        print("** Step %d updated group structure is: %s \n" %
              (ctr, newGroup.getPartitions()))

        # print "covariate pool: ", covariates_pool
        # print "best covariate index so far: ", bestCovariateIndex

        if bestCovariateIndex in covariates_pool:
            covariates_pool.remove(bestCovariateIndex)
            oldGroup = newGroup
            if ctr == p-1:
                print("** Finish with complete iterations. ** \n")
        else:
            print("** Finish with early termination at step %d "
                  "due to no further improvement of R2. ** \n" % ctr)
            break

    # print ( "** SELECTED GROUP STRUCTURE: %s with R2 = %f ** \n" %
    #         (oldGroup.getPartitions(), bestR2) )
    print ("** SELECTED GROUP STRUCTURE: %s with R2 (penalized) = %f ** \n" %
           (oldGroup.getPartitions(), bestR2))
    # return dict(group=oldGroup, r2=bestR2)
    return dict(group=oldGroup, r2=bestR2)


def backwardPartitionWithKnownResponse(data, kernel, g, method='vanilla',
                                       rank=10, seed=None, mu=1e-4, alpha=np.e):
    '''

    :param data:
    :param kernel:
    :param g: response vector
    :param method:
    :param rank:
    :param seed:
    :param mu: the first tuning parameter for the capacity penalty $\lambda * a^d$
    :param alpha: the second tuning parameter for capacity penalty in $\lambda * a^d$
    :return:
    '''
    # TODO: need to train the most general group structure ?
    # todo: Seems it is not trained now.

    covariates_pool = list(np.arange(data.p) + 1)
    oldGroup = Group(covariates_pool)  # start with a large single group
    newGroup = oldGroup
    # fit the group structure, and calculate
    #   the penalized R2
    print("** === Step 0 === **")
    print("Initial group structure: %s" % (oldGroup.getPartitions(),))
    currentOKGT = OKGTReg2(data, kernel=kernel, group=oldGroup)
    keep_res = currentOKGT._train_lr(g)
    capacity = rkhsCapacity(oldGroup, alpha)
    bestR2 = keep_res['r2'] - mu * capacity
    print("\t initial R2 (penalized) =\t\t %.10f \n" % bestR2)
    p = oldGroup.p
    # bestR2 = 0.
    # bestR2 = -np.inf
    bestCovariateIndex = None
    # keep_res = None
    ctr = 0

    while len(covariates_pool) > 1:
        ctr += 1
        print("** === Step %d === **" % ctr)
        # Create a new group
        print("** Create a new group: **")
        for covariateInd in covariates_pool:
            print("\t Create a new group for covariate %d ..." % covariateInd)
            _currentGroup = oldGroup.removeOneCovariate(covariateInd)
            currentGroup = _currentGroup.addNewCovariateAsGroup(covariateInd)
            # print("\t\t current group structure: %s " % (currentGroup.getPartitions(),))
            # Contrary to forward selection, the data matrix doesn't
            # change.
            # parameters = Parameters(currentGroup, kernel, [kernel]*currentGroup.size)
            # currentOKGT = OKGTReg2(data, parameters)
            currentOKGT = OKGTReg2(data, kernel=kernel, group=currentGroup)
            # Train OKGT
            # res = currentOKGT._train_Vanilla2(g)
            res = currentOKGT._train_lr(g)
            capacity = rkhsCapacity(currentGroup, alpha)
            print("\t\t current group structure: %s with capacity: %.04f" %
                  (currentGroup.getPartitions(), capacity))
            currentR2 = res['r2'] - mu * capacity
            if currentR2 > bestR2:
                # print("\t\t current R2 =\t %.10f \t *" % currentR2)
                print("\t\t current R2 (penalized) =\t %.10f \t *" % currentR2)
                bestR2 = currentR2
                newGroup = currentGroup
                bestCovariateIndex = covariateInd
                keep_res = res
            else:
                # print("\t\t current R2 =\t %.10f" % currentR2)
                print("\t\t current R2 (penalized) =\t %.10f" % currentR2)

            # print("\t\t best R2 =\t\t %.10f \n" % bestR2)
            print("\t\t best R2 (penalized) =\t\t %.10f \n" % bestR2)

        # print("** Updated group structure is: %s \n" % (newGroup.partition, ))
        # print '\n'
        # If there are already new groups, a chosen variable can join one of the
        #   new groups instead of creating a new group.
        print "** Add to an existing group: **"
        if oldGroup.size > 1:
            for covariateInd in covariates_pool:
                print("\t try adding covariate %d " % covariateInd)
                # Remove `covariateInd`-th covariate from the pool,
                #   which will be added into one of the other groups.
                updatedCovariatesPool = copy.deepcopy(covariates_pool)
                updatedCovariatesPool.remove(covariateInd)
                # Get the group number of the chosen `covariateInd`
                covariateMember = oldGroup.getMembership(covariateInd)
                # Take all other groups as a Group object
                otherGroupInds = list(np.arange(oldGroup.size) + 1)
                otherGroupInds.remove(covariateMember)

                # print type(otherGroupInds), ": ", otherGroupInds

                otherGroup = oldGroup.getPartitions(otherGroupInds, True)
                # Try adding the chosen `covariateInd` to each of the other groups
                for groupInd in np.arange(otherGroup.size) + 1:
                    print("\t   in other group %d ..." % groupInd)
                    updatedOtherGroup = otherGroup.addNewCovariateToGroup(covariateInd, groupInd)
                    currentGroup = updatedOtherGroup + updatedCovariatesPool
                    print("\t\t current group structure: %s " % (currentGroup.getPartitions(),))
                    # parameters = Parameters(currentGroup, kernel, [kernel]*currentGroup.size)
                    # currentOKGT = OKGTReg2(data, parameters)
                    currentOKGT = OKGTReg2(data, kernel=kernel, group=currentGroup)
                    # Train OKGT
                    # res = currentOKGT.train(method, rank, seed)
                    # res = currentOKGT._train_Vanilla2(g)
                    res = currentOKGT._train_lr(g)
                    # currentR2 = res['r2']
                    # capacity = sum([len(g) ** len(g) for g in currentGroup.partition])
                    capacity = rkhsCapacity(currentGroup, alpha)
                    print("\t\t current group structure: %s with capacity: %.04f" %
                          (currentGroup.getPartitions(), capacity))
                    currentR2 = res['r2'] - mu * capacity
                    # Check if there is improvement
                    if currentR2 > bestR2:
                        # print("\t\t current R2 =\t %.10f \t *" % currentR2)
                        print("\t\t current R2 (penalized) =\t %.10f \t *" % currentR2)
                        bestR2 = currentR2
                        newGroup = currentGroup
                        bestCovariateIndex = covariateInd
                        keep_res = res
                    else:
                        # print("\t\t current R2 =\t %.10f" % currentR2)
                        print("\t\t current R2 (penalized) =\t %.10f" % currentR2)

                    # print("\t\t best R2 =\t\t %.10f \n" % bestR2)
                    print("\t\t best R2 (penalized) =\t\t %.10f \n" % bestR2)
        else:
            print("\t ** No other groups than the pool. Pass ... ** \n")

        print("** Step %d updated group structure is: %s \n" %
              (ctr, newGroup.getPartitions()))

        # print "covariate pool: ", covariates_pool
        # print "best covariate index so far: ", bestCovariateIndex

        # If the best covariate index is updated,
        #   then it should be removed from the pool
        if bestCovariateIndex in covariates_pool:
            covariates_pool.remove(bestCovariateIndex)
            oldGroup = newGroup
            if ctr == p - 1:
                print("** Finish with complete iterations. ** \n")
        else:
            print("** Finish with early termination at step %d "
                  "due to no further improvement of R2. ** \n" % ctr)
            break

    # print ( "** SELECTED GROUP STRUCTURE: %s with R2 = %f ** \n" %
    #         (oldGroup.getPartitions(), bestR2) )
    print ("** SELECTED GROUP STRUCTURE: %s with R2 (penalized) = %f ** \n" %
           (oldGroup.getPartitions(), bestR2))
    # return dict(group=oldGroup, r2=bestR2)
    # return dict(group=oldGroup, r2=bestR2)
    return dict(group=oldGroup, r2=bestR2, f_call=keep_res['f_call'])


if __name__ == "__main__":
    # from okgtreg.DataSimulator import DataSimulator
    from okgtreg.Kernel import Kernel
    from okgtreg.Data import Data


    # # Simulate data
    # np.random.seed(25)
    # # y, x = DataSimulator.SimData_Wang04(500)
    # y, x = DataSimulator.SimData_Wang04WithInteraction(500)
    # data = Data(y, x)
    #
    # # Same kernel for all groups
    # kernel = Kernel('gaussian', sigma=0.5)
    #
    # # Call backward selection
    # selectedGroup = backwardPartition(data, kernel, 'nystroem', 10)

    # Model
    def model10(n):
        x = np.random.uniform(1, 2, (n, 6))
        e = np.random.normal(0., 0.1, n)
        g = 1. + \
            np.log(x[:, 0]) + \
            x[:, 1] / np.exp(x[:, 2]) + \
            np.power(x[:, 3] + x[:, 4], x[:, 5]) + e
        y = np.log(g)
        return Data(y, x), Group([1], [2, 3], [4, 5, 6]), g


    # Simulate data
    n = 500
    np.random.seed(25)
    data, tgroup, g = model10(500)

    print tgroup

    # Run forward inclusion method for group structure detection
    kernel = Kernel('gaussian', sigma=0.5)
    res = backwardPartition(data, kernel)

    # With g known
    res = backwardPartitionWithKnownResponse(data, kernel, g)
