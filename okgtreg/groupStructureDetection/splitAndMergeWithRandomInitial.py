"""
This simulation illustrate step-by-step the procedure of group structure determination
proposed by Michael.

We start with a random partition of the predictor variables. The corresponding
OKGT is fitted with R2 being recorded. Then we perform the following split and
merge operators until there is no further improvement in R2:

1. For each group of size > 1, split it into individual covariates and fit
   the resulting OKGT and record its R2.

   Compare R2 under each scenario in (2) with the R2 we started with. Pick the
   the scenario with gives the largest improvement in R2 and randomly pick a
   covariate from the group to form a uni-variate group.

2. For each univariate group in the current structure, try to merge its covariate
   with one of the other groups, regardless of the other group being univariate or
   multivariate.

Then, step 1 and 2 are repeated iteratively until no further improvement in R2.

Can be extended to include removing steps for variable selection.

Question:

1. If different random partitions, do they result in the same optimal group structure?
"""

import numpy as np

from okgtreg.Group import RandomGroup
from okgtreg.OKGTReg import OKGTRegForDetermineGroupStructure
from okgtreg.Parameters import Parameters


def splitAndMergeWithRandomInitial(data, kernel, useLowRankApproximation=True, rank=10, seed=None):
    """
    with aggressive split.

    :type seed: int
    :param seed: seed for random group initialization and Nystroem method
                 of low rank approximation

    :param data:
    :param kernel:
    :param useLowRankApproximation:
    :param rank:

    :rtype: OKGTRegForDetermineGroupStructure
    :return:
    """
    method = 'nystroem' if useLowRankApproximation else 'vanilla'

    # np.random.seed(seed)
    # group0 = RandomGroup(4, [i+1 for i in range(data.p)])
    group0 = RandomGroup(4, nCovariates=data.p, seed=seed)
    parameters0 = Parameters(group0, kernel, [kernel]*group0.size)
    okgt = OKGTRegForDetermineGroupStructure(data, parameters0)

    proceed = True
    counter = 0
    while proceed:
        counter += 1
        print("\n=== %d ===" % counter)

        print "[Split]"
        okgtAfterSplit = okgt.optimalSplit(kernel, method, rank, seed)
        print "[Merge]"
        okgtAfterMerge = okgt.optimalMerge(kernel, method, rank, seed)

        if okgtAfterSplit.r2 == okgtAfterMerge.r2:
            proceed = False
        elif okgtAfterSplit.r2 > okgtAfterMerge.r2:
            okgt = okgtAfterSplit
        else:
            okgt = okgtAfterMerge
        print("\n**** Updated group structure: %s, %.04f. ****\n" % (okgt.getGroupStructure(), okgt.r2))

    print("\n=== Final ===")
    print(">>>> Best group structure: %s." % okgt.getGroupStructure())
    return okgt

def splitAndMergeWithRandomInitial2(data, kernel, useLowRankApproximation=True, rank=10, seed=None):
    """
    Less aggressive verison.

    :param seed:
    :param data:
    :param kernel:
    :param useLowRankApproximation:
    :param rank:
    :return:
    """
    method = 'nystroem' if useLowRankApproximation else 'vanilla'

    # Random group initialization
    group0 = RandomGroup(4, nCovariates=data.p, seed=seed)
    parameters0 = Parameters(group0, kernel, [kernel]*group0.size)
    okgt = OKGTRegForDetermineGroupStructure(data, parameters0)

    # Start group structure detection procedure
    proceed = True
    counter = 0
    while proceed:
        counter += 1
        print("\n=== %d ===" % counter)

        print "[Split]"
        okgtAfterSplit = okgt.optimalSplit2(kernel, method, rank, seed)  # less aggressive split
        print "[Merge]"
        okgtAfterMerge = okgt.optimalMerge(kernel, method, rank, seed)

        if okgtAfterSplit.r2 == okgtAfterMerge.r2:  # no split or merge can improve fitting
            proceed = False
        elif okgtAfterSplit.r2 > okgtAfterMerge.r2:
            okgt = okgtAfterSplit
        else:
            okgt = okgtAfterMerge
        print("\n**** Updated group structure after split and merge: "
              "%s, R2 = %.04f. ****\n" % (okgt.getGroupStructure(), okgt.r2))

    print("\n=== Final ===")
    print(">>>> Best group structure: %s." % okgt.getGroupStructure())
    return okgt


if __name__=='__main__':
    import numpy as np

    from okgtreg.DataSimulator import DataSimulator
    from okgtreg.Kernel import Kernel

    # Simulate data
    n = 500
    np.random.seed(25)
    data = DataSimulator.SimData_Wang04WithInteraction2(n)
    kernel = Kernel('gaussian', sigma=0.5)

    # Aggressive split
    print("=== Aggressive split ===")
    optimalOkgt = splitAndMergeWithRandomInitial(data, kernel, seed=25)
    print optimalOkgt.getGroupStructure(), '\n'
    #
    # # -----------------
    # from okgtreg.Group import Group
    # from okgtreg.OKGTReg import OKGTReg
    #
    # trueGroup = Group([1], [2], [3], [4], [5], [6,7,8])
    # trueParameters = Parameters(trueGroup, kernel, [kernel]*trueGroup.size)
    # trueOkgt = OKGTReg(data, trueParameters)
    # fit = trueOkgt.train('nystroem', 10, 25)
    # fit['r2']
    # # -----------------

    # Less aggressive split
    print "=== Less aggressive split ==="
    optimalOkgt2 = splitAndMergeWithRandomInitial2(data, kernel, seed=25)
    print optimalOkgt2.getGroupStructure(), '\n'
